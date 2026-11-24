"""従来の OTF 観測(LO 固定・OFF 点を引く)の模擬観測と処理(FMLO との比較用)

FMLO と同じスキャン軌跡(scan.ScanPass)・同じ正解キューブ・同じ雑音モデルを使い、
次の点だけを変える:
  - LO は変調しない(全時刻でシフト 0。観測窓は変調なしの窓)
  - otf.off_interval_rows 行ごとに、天体のない OFF 点を otf.off_integration_s 秒観測する
    (最初の行の前と最後の行の後にも OFF を置き、各行が前後の OFF にはさまれるようにする)
  - 処理は ON − OFF(前後の OFF を時間で補間)→ 輝線のないチャンネルで多項式ベースラインを引く

雑音は OFF の時間も含めた 1 本の時系列として作る(大気の時間変動が ON と OFF の差として残る)。

簡略化していること:
  - ON と OFF のあいだの移動時間・行の折り返し時間は 0
  - 強度は加算的な K 単位のまま。チョッパーホイール較正の (ON − OFF)/OFF × T_sys は使わない
    (利得変動は模型に入っていないので、差を取るだけと同じ)
  - OFF 点には天体の放射がない
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cube as cube_mod
from . import modulation, noise, observation
from .config import Config, spawn_rngs
from .scan import ScanPass

# FMLO のパス(spawn_rngs(seed, i))と重ならない乱数系列にするための pass_index のオフセット
OTF_PASS_INDEX_OFFSET = 100


@dataclass
class OtfTimeline:
    """OFF を含めた全時刻の並び"""

    time_s: np.ndarray        # (n_total,) [s]
    on_index: np.ndarray      # (n_total,) ON なら scan_pass の何番目のサンプルか。OFF は -1
    off_block: np.ndarray     # (n_total,) OFF なら何番目の OFF ブロックか。ON は -1

    @property
    def n_total(self) -> int:
        return self.time_s.size

    @property
    def n_off_blocks(self) -> int:
        return int(self.off_block.max()) + 1


def make_timeline(scan_pass: ScanPass, off_interval_rows: int, n_off: int, dt_s: float) -> OtfTimeline:
    """OFF → ON(off_interval_rows 行)→ OFF → … → ON → OFF の順に並べる"""
    if n_off < 1:
        raise ValueError("OFF のサンプル数は 1 以上です")
    rows = np.asarray(scan_pass.row)
    if np.any(np.diff(rows) < 0):
        raise ValueError("scan_pass の行番号が時間順に並んでいません")
    n_rows = int(rows.max()) + 1
    on_parts, off_parts = [], []
    block = 0
    for g in range(0, n_rows, off_interval_rows):
        on_parts.append(np.full(n_off, -1))
        off_parts.append(np.full(n_off, block))
        block += 1
        idx = np.flatnonzero((rows >= g) & (rows < g + off_interval_rows))
        on_parts.append(idx)
        off_parts.append(np.full(idx.size, -1))
    on_parts.append(np.full(n_off, -1))
    off_parts.append(np.full(n_off, block))
    on_index = np.concatenate(on_parts)
    off_block = np.concatenate(off_parts)
    return OtfTimeline(np.arange(on_index.size) * dt_s, on_index, off_block)


def off_weights(timeline: OtfTimeline, method="linear"):
    """各 ON サンプルで OFF スペクトルを補間する重み

    Returns (b0, b1, w1): OFF = (1 − w1) × OFF[b0] + w1 × OFF[b1]。ON サンプルの順(scan_pass の順)
    """
    tl = timeline
    is_off = tl.off_block >= 0
    t_blk = np.array([tl.time_s[tl.off_block == b].mean() for b in range(tl.n_off_blocks)])
    on_pos = np.flatnonzero(~is_off)
    order = np.argsort(tl.on_index[on_pos])
    t_on = tl.time_s[on_pos][order]
    b0 = np.searchsorted(t_blk, t_on) - 1
    if np.any(b0 < 0) or np.any(b0 + 1 >= t_blk.size):
        raise ValueError("前後を OFF にはさまれていない ON サンプルがあります")
    b1 = b0 + 1
    w1 = (t_on - t_blk[b0]) / (t_blk[b1] - t_blk[b0])
    if method == "nearest":
        w1 = (w1 >= 0.5).astype(float)
    elif method != "linear":
        raise ValueError(f"未知の OFF の補間方法です: {method}")
    return b0, b1, w1


def polynomial_baseline(spectra, free_mask, order):
    """各スペクトルから、free_mask のチャンネルで最小二乗で合わせた order 次の多項式を引く

    spectra : (n, n_channels)。order < 0 なら何もしない
    """
    spectra = np.asarray(spectra, dtype=float)
    if order < 0:
        return spectra.copy(), np.zeros_like(spectra)
    n_ch = spectra.shape[1]
    if free_mask.sum() <= order:
        raise ValueError("ベースラインを合わせる輝線のないチャンネルが足りません")
    x = np.linspace(-1.0, 1.0, n_ch)
    a = np.vander(x, order + 1, increasing=True)                  # (n_ch, order+1)
    coef = np.linalg.lstsq(a[free_mask], spectra[:, free_mask].T, rcond=None)[0]
    base = (a @ coef).T
    return spectra - base, base


@dataclass
class OtfObservation:
    """1 パス分の従来 OTF 観測。signal_on / noise は正解情報(評価専用)"""

    scan_pass: ScanPass
    grid: cube_mod.FrequencyGrid
    timeline: OtfTimeline
    signal_on: np.ndarray       # ON サンプルの天体信号のみ (n_on, n_channels) [K]
    noise: noise.NoiseComponents
    raw: np.ndarray             # OFF も含めた観測時系列 (n_total, n_channels) [K]

    @property
    def shifts_ch(self) -> np.ndarray:
        """ON サンプルのシフト量(全て 0)。グリッディング関数にそのまま渡せる"""
        return np.zeros(self.scan_pass.n_time, dtype=int)


def observe_pass(sky: observation.SkyCube, cfg: Config, scan_pass: ScanPass, pass_index: int) -> OtfObservation:
    """scan_pass の軌跡で、LO を固定し OFF 点をはさみながら観測した時系列を作る"""
    observation.validate_sky_cube(sky)
    grid = observation.match_frequency_grid(sky.freq_ghz, cfg.spectrometer)
    dt = cfg.modulation.dt_s
    n_off = max(1, int(round(cfg.otf.off_integration_s / dt)))
    tl = make_timeline(scan_pass, cfg.otf.off_interval_rows, n_off, dt)
    rngs = spawn_rngs(cfg.seed, OTF_PASS_INDEX_OFFSET + pass_index)
    n_ch = grid.n_channels

    wide = cube_mod.sample_spectra(sky.data, sky.ra_arcsec, sky.dec_arcsec,
                                   scan_pass.ra_arcsec, scan_pass.dec_arcsec)
    signal_on = modulation.modulate(wide, np.zeros(scan_pass.n_time, dtype=int), grid.start0, n_ch)
    del wide

    comps = noise.make_noise_components(cfg.noise, cfg.spectrometer, grid, np.zeros(tl.n_total, dtype=int),
                                        dt, rngs, external_key=f"atmosphere_otf_{scan_pass.name}")
    raw = comps.total()
    on_pos = np.flatnonzero(tl.on_index >= 0)
    raw[on_pos] += signal_on[tl.on_index[on_pos]]
    return OtfObservation(scan_pass, grid, tl, signal_on, comps, raw)


@dataclass
class OtfReduction:
    off_spectra: np.ndarray     # (n_off_blocks, n_channels) [K]
    on_minus_off: np.ndarray    # (n_on, n_channels) [K] ベースラインを引く前
    calibrated: np.ndarray      # (n_on, n_channels) [K] ベースラインを引いた後
    baseline: np.ndarray        # (n_on, n_channels) [K]


def reduce_pass(obs: OtfObservation, line_free_channels: np.ndarray, off_interpolation="linear",
                baseline_order=1) -> OtfReduction:
    """ON − OFF(補間)→ 多項式ベースラインを引く。正解は使わない

    line_free_channels : (n_channels,) ベースラインを合わせるチャンネル(観測者が輝線の位置から決める)
    """
    tl = obs.timeline
    off = np.stack([obs.raw[tl.off_block == b].mean(axis=0) for b in range(tl.n_off_blocks)])
    on_pos = np.flatnonzero(tl.on_index >= 0)
    on = np.empty((obs.scan_pass.n_time, obs.raw.shape[1]))
    on[tl.on_index[on_pos]] = obs.raw[on_pos]
    b0, b1, w1 = off_weights(tl, off_interpolation)
    diff = on - ((1.0 - w1)[:, None] * off[b0] + w1[:, None] * off[b1])
    cal, base = polynomial_baseline(diff, np.asarray(line_free_channels, dtype=bool), baseline_order)
    return OtfReduction(off, diff, cal, base)


def time_summary(obs: OtfObservation, dt_s: float) -> dict:
    tl = obs.timeline
    n_on = int((tl.on_index >= 0).sum())
    return {"name": obs.scan_pass.name, "n_off_blocks": tl.n_off_blocks,
            "off_samples_per_block": int((tl.off_block == 0).sum()),
            "on_time_s": n_on * dt_s, "off_time_s": (tl.n_total - n_on) * dt_s,
            "total_time_s": tl.n_total * dt_s}
