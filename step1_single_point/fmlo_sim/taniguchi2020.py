"""Taniguchi et al. (2020, PASJ 72, 2) Fig. 6 の再現シミュレーター

論文 §2(原理)と §3.2(データ解析手順)の式をそのまま実装し、Fig. 6
(反復アルゴリズムのフローチャート:T^cal, T^cor, T^ast, E^nc の 2 反復分)
を出力するための模擬データ生成と解析を行う。

既存の 150 GHz 想定パイプライン(pipeline.py / iterative.py)とは独立している。

配列の向き
----------
論文は timestream を D 行 × N 列(周波数 × 時間)で書くが、本モジュールは
numpy で扱いやすい (N, D) = (時間, チャンネル) で保持する(FMFlow と同じ向き)。

各処理の根拠
------------
- 変調・復調 M, M^-1 ……………… 論文 §2.2 の式(NaN を含む写像)
- 相関成分の推定 (PCA) ……………… 論文 §2.5。時間平均を引いて PCA し、
  平均は戻す(FMFlow v0.2.7 `fm.models.pca(centering=True)` と同じ)
- スペクトルと雑音 ……………………… 論文 §2.6 の重み付き平均。雑音は
  FMFlow v0.2.7 `tospectrum` と同じ sqrt((<X^2>-<X>^2)/n)
- σ-cutoff ……………………………………… 論文 §2.7。Fig. 6 の T^ast パネルに負値が
  無いことから、FMFlow `AstroLines(function='cutoff')` と同じ片側判定
  (s/σ < θ → 0)を既定とする。`two_sided=True` で論文式の |s| 判定
- 反復 …………………………………………… 論文 §3.2 の手順 1–5(イメージ側なし)
- 収束判定 ……………………………………… FMFlow `Convergence` と同じ
  |T_new − T_old|_F / |T_old|_F < ε
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent  # step1_single_point/


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
@dataclass
class SpectrometerCfg:
    n_channels: int = 2048            # D
    channel_width_mhz: float = 0.9765625
    n_time: int = 100                 # N
    dt_s: float = 0.1


@dataclass
class FmCfg:
    step_ch: int = 23                 # FM step [ch / sample]
    n_steps: int = 24                 # FM width = n_steps * step_ch [ch]
    phase: int = 36                   # 先頭サンプルが基本三角列のどこから始まるか


@dataclass
class LineCfg:
    amplitude: float = 5.0            # 単位: 論文図の E^nc パネルの標準偏差(config の注記参照)
    center_ch: float = 1321.2         # 復調後 (RF) チャンネル番号 (= ξ̄=0 のときの IF ch)
    fwhm_ch: float = 36.52            # ガウシアンの FWHM [ch]


@dataclass
class CorrelatedCfg:
    profile: str = "digitized"        # digitized | constant
    profile_file: str = "data/taniguchi2020_fig6_correlated_profile.csv"
    constant_level: float = 0.0       # profile=constant のときの値
    scale: float = 1.0                # 時間プロファイルに掛ける係数


@dataclass
class NoiseCfg:
    sigma: float = 1.06


@dataclass
class ReductionCfg:
    n_components: List[int] = field(default_factory=lambda: [4, 1])  # 反復ごとの K(最後の値を以降も使う)
    centering: bool = True
    cutoff_sigma: float = 5.0         # θ_cutoff
    two_sided_cutoff: bool = False
    n_iterations: int = 2             # 図に描く反復数(Fig. 6 は 2)
    convergence_eps: Optional[float] = None  # 指定すると収束まで反復 (論文 §3.2: 0.05)
    max_iterations: int = 50


@dataclass
class PlotCfg:
    vmin: float = -1.67               # 全パネル共通のカラーレンジ(論文図から読み取り)
    vmax: float = 5.68
    cmap: str = "viridis"
    panel_px: int = 720               # 論文 PDF に埋め込まれたパネル画像の画素数


@dataclass
class Fig6Config:
    seed: int = 2020
    spectrometer: SpectrometerCfg = field(default_factory=SpectrometerCfg)
    fm: FmCfg = field(default_factory=FmCfg)
    line: LineCfg = field(default_factory=LineCfg)
    correlated: CorrelatedCfg = field(default_factory=CorrelatedCfg)
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    reduction: ReductionCfg = field(default_factory=ReductionCfg)
    plot: PlotCfg = field(default_factory=PlotCfg)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {
    "spectrometer": SpectrometerCfg, "fm": FmCfg, "line": LineCfg,
    "correlated": CorrelatedCfg, "noise": NoiseCfg, "reduction": ReductionCfg,
    "plot": PlotCfg,
}


def _coerce(value, current):
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, list):
        if isinstance(value, str):
            return [int(v) for v in value.strip("[]").split(",") if v.strip()]
        return [int(v) for v in value]
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        if value in (None, "null", "None"):
            return None
        return float(value)
    return value


def load_fig6_config(path=None, overrides: Optional[dict] = None) -> Fig6Config:
    """YAML(省略時は既定値)を読み、`section.key=value` 形式の上書きを適用する"""
    data = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    cfg = Fig6Config()
    unknown = set(data) - set(_SECTIONS) - {"seed"}
    if unknown:
        raise ValueError(f"未知の設定セクション: {sorted(unknown)}")
    if "seed" in data:
        cfg.seed = int(data["seed"])
    for name, cls in _SECTIONS.items():
        sec = data.get(name) or {}
        obj = getattr(cfg, name)
        for k, v in sec.items():
            if not hasattr(obj, k):
                raise ValueError(f"未知の設定キー: {name}.{k}")
            setattr(obj, k, _coerce(v, getattr(obj, k)))
    for key, v in (overrides or {}).items():
        if key == "seed":
            cfg.seed = int(v)
            continue
        sec, _, k = key.partition(".")
        if sec not in _SECTIONS or not hasattr(getattr(cfg, sec), k):
            raise ValueError(f"未知の設定キー: {key}")
        obj = getattr(cfg, sec)
        setattr(obj, k, _coerce(v, getattr(obj, k)))
    validate(cfg)
    return cfg


def validate(cfg: Fig6Config) -> None:
    s, fm, r = cfg.spectrometer, cfg.fm, cfg.reduction
    if s.n_channels < 1 or s.n_time < 3:
        raise ValueError("n_channels >= 1, n_time >= 3 が必要です")
    if fm.step_ch < 0 or fm.n_steps < 0:
        raise ValueError("fm.step_ch, fm.n_steps は 0 以上")
    if not r.n_components or any(k < 0 for k in r.n_components):
        raise ValueError("reduction.n_components は 0 以上の整数のリスト")
    if any(k >= min(s.n_time, s.n_channels) for k in r.n_components):
        raise ValueError("K < min(N, D) が必要です(論文 §2.5)")
    if r.n_iterations < 1:
        raise ValueError("reduction.n_iterations >= 1")
    if cfg.correlated.profile not in ("digitized", "constant"):
        raise ValueError("correlated.profile は digitized / constant")
    if cfg.plot.vmin >= cfg.plot.vmax:
        raise ValueError("plot.vmin < plot.vmax")


# ---------------------------------------------------------------------------
# FM パターン・変調・復調 (論文 §2.2)
# ---------------------------------------------------------------------------
def zigzag_fm_channels(n_time: int, step_ch: int, n_steps: int, phase: int = 0) -> np.ndarray:
    """ゼロ基準の FM channel ξ̄(n)(論文 式 blfmch)を返す

    基本列は 0, 1, ..., n_steps, n_steps, ..., 1, 0(周期 2(n_steps+1) サンプル。
    折り返し点の値が 2 サンプル続く)に step_ch を掛けたもの。Fig. 6 のパネルから
    読み取った形(周期 50 サンプル、折り返し点で同じ ch が 2 回)と一致する。
    """
    base = np.r_[np.arange(n_steps + 1), np.arange(n_steps + 1)[::-1]]
    idx = (np.arange(n_time) + phase) % base.size
    xi = step_ch * base[idx]
    return (xi - xi.min()).astype(int)


def n_demod_channels(n_channels: int, xi_bar: np.ndarray) -> int:
    """D~ = D + max(ξ̄)"""
    return int(n_channels + np.max(xi_bar))


def _index(n_channels: int, xi_bar: np.ndarray) -> np.ndarray:
    return np.arange(n_channels)[None, :] + np.asarray(xi_bar)[:, None]


def modulate(demod: np.ndarray, xi_bar: np.ndarray, n_channels: int) -> np.ndarray:
    """M: X~ (N, D~) → X (N, D)。X_{n,d} = X~_{n, d+ξ̄(n)}"""
    demod = np.asarray(demod)
    rows = np.arange(demod.shape[0])[:, None]
    return demod[rows, _index(n_channels, xi_bar)]


def demodulate(mod: np.ndarray, xi_bar: np.ndarray) -> np.ndarray:
    """M^-1: X (N, D) → X~ (N, D~)。観測されない要素は NaN"""
    mod = np.asarray(mod, dtype=float)
    n_time, n_channels = mod.shape
    out = np.full((n_time, n_demod_channels(n_channels, xi_bar)), np.nan)
    out[np.arange(n_time)[:, None], _index(n_channels, xi_bar)] = mod
    return out


# ---------------------------------------------------------------------------
# 解析の各要素
# ---------------------------------------------------------------------------
def pca_correlated(x: np.ndarray, n_components: int, centering: bool = True) -> np.ndarray:
    """相関成分 X^c = P[:, :K] @ C[:K] (+ 時間平均) を返す(論文 §2.5、FMFlow pca)"""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0) if centering else np.zeros(x.shape[1])
    if n_components == 0:
        return np.broadcast_to(mean, x.shape).copy()
    u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = int(n_components)
    return (u[:, :k] * s[:k]) @ vt[:k] + mean


def make_spectrum(demod: np.ndarray):
    """時間方向の NaN 除外平均スペクトルと雑音(FMFlow tospectrum と同じ定義)

    Returns: spectrum (D~,), noise (D~,), hits (D~,)
    """
    w = ~np.isnan(demod)
    hits = w.sum(axis=0)
    x = np.where(w, demod, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = x.sum(axis=0) / hits
        m2 = (x ** 2).sum(axis=0) / hits
        noise = np.sqrt(np.clip(m2 - m1 ** 2, 0.0, None) / hits)
    noise[hits <= 2] = np.inf
    m1[hits == 0] = 0.0
    return m1, noise, hits


def sigma_cutoff(spectrum: np.ndarray, noise: np.ndarray, theta: float, two_sided: bool = False):
    """σ-cutoff による雑音なしモデル(論文 §2.7)"""
    snr = np.abs(spectrum) / noise if two_sided else spectrum / noise
    return np.where(snr > theta, spectrum, 0.0)


# ---------------------------------------------------------------------------
# 模擬データ
# ---------------------------------------------------------------------------
@dataclass
class Fig6Truth:
    xi_bar: np.ndarray        # (N,)
    t_cal: np.ndarray         # (N, D)  観測 timestream T^cal
    t_cor: np.ndarray         # (N, D)  真の相関成分
    t_ast: np.ndarray         # (N, D)  真の天体輝線(変調済み)
    e_nc: np.ndarray          # (N, D)  非相関雑音
    line_spectrum: np.ndarray  # (D~,) 真の輝線スペクトル
    correlated_profile: np.ndarray  # (N,)


def load_correlated_profile(cfg: Fig6Config) -> np.ndarray:
    n = cfg.spectrometer.n_time
    c = cfg.correlated
    if c.profile == "constant":
        prof = np.full(n, float(c.constant_level))
    else:
        path = Path(c.profile_file)
        if not path.is_absolute():
            path = HERE / path
        prof = np.loadtxt(path, delimiter=",", comments="#", usecols=1, encoding="utf-8")
        if prof.size != n:
            raise ValueError(
                f"相関成分プロファイルの長さ {prof.size} が n_time={n} と一致しません({path})"
            )
    return c.scale * prof


def simulate_fig6(cfg: Fig6Config) -> Fig6Truth:
    """T^cal = T^cor + T^ast + E^nc(論文 式 simple-components、イメージ側なし)"""
    sp, fm, ln = cfg.spectrometer, cfg.fm, cfg.line
    rng = np.random.default_rng(cfg.seed)
    xi_bar = zigzag_fm_channels(sp.n_time, fm.step_ch, fm.n_steps, fm.phase)
    d_tilde = n_demod_channels(sp.n_channels, xi_bar)

    ch = np.arange(d_tilde)
    line = ln.amplitude * np.exp(-4.0 * np.log(2.0) * (ch - ln.center_ch) ** 2 / ln.fwhm_ch ** 2)
    t_ast = modulate(np.broadcast_to(line, (sp.n_time, d_tilde)), xi_bar, sp.n_channels)

    prof = load_correlated_profile(cfg)
    t_cor = np.broadcast_to(prof[:, None], (sp.n_time, sp.n_channels)).copy()

    e_nc = cfg.noise.sigma * rng.standard_normal((sp.n_time, sp.n_channels))
    return Fig6Truth(
        xi_bar=xi_bar, t_cal=t_cor + t_ast + e_nc, t_cor=t_cor, t_ast=t_ast,
        e_nc=e_nc, line_spectrum=line, correlated_profile=prof,
    )


# ---------------------------------------------------------------------------
# 反復アルゴリズム (論文 §3.2)
# ---------------------------------------------------------------------------
@dataclass
class IterationStep:
    n_components: int
    x_in: np.ndarray          # PCA に入れた timestream: T^cal − T^ast_{r−1}
    t_cor: np.ndarray         # T^cor_r の推定
    t_ast: np.ndarray         # T^ast_r の推定(変調済み)
    e_nc: np.ndarray          # 残差 T^cal − T^cor_r − T^ast_r
    spectrum: np.ndarray      # 復調スペクトル s~
    noise: np.ndarray         # s~ の雑音
    model: np.ndarray         # σ-cutoff モデル
    variation: Optional[float]  # |T^cln_r − T^cln_{r−1}|_F / |T^cln_{r−1}|_F


@dataclass
class Fig6Result:
    steps: List[IterationStep]
    hits: np.ndarray
    converged: Optional[bool]


def run_iterations(t_cal: np.ndarray, xi_bar: np.ndarray, n_components: Sequence[int],
                   centering: bool = True, cutoff_sigma: float = 5.0,
                   two_sided_cutoff: bool = False, n_iterations: int = 2,
                   convergence_eps: Optional[float] = None,
                   max_iterations: int = 50) -> Fig6Result:
    """論文 §3.2 の反復(イメージ側の輝線なし)

    1. T^ast = 0
    2. T^cor = PCA_K(T^cal − T^ast)
    3. T^ast = M( cutoff( spectrum( M^-1(T^cal − T^cor) ) ) )
    5. T^cln = T^cal − T^cor。convergence_eps を与えた場合は収束まで 2 へ戻る。
       与えない場合は n_iterations 回で終了。
    """
    t_cal = np.asarray(t_cal, dtype=float)
    n_channels = t_cal.shape[1]
    total = max_iterations if convergence_eps is not None else n_iterations
    t_ast = np.zeros_like(t_cal)
    steps: List[IterationStep] = []
    prev_cln = None
    converged = None
    hits = None
    for r in range(total):
        k = int(n_components[min(r, len(n_components) - 1)])
        x_in = t_cal - t_ast
        t_cor = pca_correlated(x_in, k, centering)
        t_cln = t_cal - t_cor
        spec, noise, hits = make_spectrum(demodulate(t_cln, xi_bar))
        model = sigma_cutoff(spec, noise, cutoff_sigma, two_sided_cutoff)
        t_ast = modulate(np.broadcast_to(model, (t_cal.shape[0], model.size)), xi_bar, n_channels)
        var = None
        if prev_cln is not None:
            var = float(np.linalg.norm(t_cln - prev_cln) / np.linalg.norm(prev_cln))
        steps.append(IterationStep(k, x_in, t_cor, t_ast, t_cal - t_cor - t_ast,
                                   spec, noise, model, var))
        prev_cln = t_cln
        if convergence_eps is not None and var is not None and r + 1 >= n_iterations:
            if var < convergence_eps:
                converged = True
                break
    else:
        if convergence_eps is not None:
            converged = False
    return Fig6Result(steps=steps, hits=hits, converged=converged)


def run_fig6(cfg: Fig6Config):
    truth = simulate_fig6(cfg)
    r = cfg.reduction
    result = run_iterations(
        truth.t_cal, truth.xi_bar, r.n_components, r.centering, r.cutoff_sigma,
        r.two_sided_cutoff, r.n_iterations, r.convergence_eps, r.max_iterations,
    )
    return truth, result


def to_panel_pixels(tod: np.ndarray, n_px: int = 720) -> np.ndarray:
    """(N, D) を横 n_px 画素へ最近傍で間引く(論文 PDF のパネル画像と同じ横画素数)"""
    d = tod.shape[1]
    cols = ((np.arange(n_px) + 0.5) * d / n_px).astype(int)
    return tod[:, np.clip(cols, 0, d - 1)]
