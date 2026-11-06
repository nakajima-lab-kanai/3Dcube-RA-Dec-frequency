"""Taniguchi et al. (2020) の Fig. 6 に対応する 2 ステップ成分推定

観測 TOD から相関成分 (C_r) と天体信号 (A_r) を交互に推定し、
残差 (E_r) を求める。

各記号は同一の (n_time, n_channels) 配列:
  X   : 観測 TOD
  U_r : r 回目の PCA 入力 TOD (U_1 = X, U_2 = X - A_1)
  C_r : r 回目で推定された相関成分 TOD (PCA 除去成分)
  A_r : r 回目で推定された天体信号 TOD (復調・モデル化・再変調)
  E_r : r 回目の残差 TOD (E_r = X - C_r - A_r)

重要:
  - 正解信号や真の雑音には一切依存せず、観測データ・シフト量・座標のみを入力とする。
  - 数学的恒等式 X = C_r + A_r + E_r が厳密に成り立つ。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import modulation, removal


@dataclass
class IterationResult:
    """r 回目の成分推定結果"""

    r: int
    u_tod: np.ndarray              # PCA 入力 TOD (n_time, n_channels) [K]
    c_tod: np.ndarray              # 推定相関成分 TOD (n_time, n_channels) [K]
    a_tod: np.ndarray              # 推定天体信号 TOD (n_time, n_channels) [K]
    e_tod: np.ndarray              # 残差 TOD (n_time, n_channels) [K]
    restored_spectrum: np.ndarray  # X - C_r を復調・時間平均したスペクトル (n_wide,) [K]
    model_spectrum: np.ndarray     # cutoff を適用した天体モデルスペクトル (n_wide,) [K]
    noise_sigma: np.ndarray        # 各 sky チャンネルの推定雑音水準 (n_wide,) [K]
    hits: np.ndarray               # 各 sky チャンネルの観測ヒット数 (n_wide,)
    explained_ratio: np.ndarray    # PCA 主成分の寄与率
    noise_status: str              # 雑音推定状態 ("ok" / "insufficient_points" / "zero_mad")
    noise_n_valid: int             # 雑音推定に用いられた有効点数


@dataclass
class TwoStepResult:
    """2 ステップ反復推定の全体結果"""

    x_tod: np.ndarray              # 元の観測 TOD X (n_time, n_channels) [K]
    iter1: IterationResult         # 1回目の推定
    iter2: IterationResult         # 2回目の推定


def estimate_noise_sigma(
    spectrum: np.ndarray,
    hits: np.ndarray,
    n_time: int,
) -> tuple[np.ndarray, float, str, int]:
    """復調スペクトルの全時刻観測領域からロバスト雑音水準 (MAD) を推定する (簡易推定法)

    注意:
      本手法はシミュレーション用の簡易推定法であり、Taniguchi et al. (2020) の
      重み付き分散推定と同一ではありません。

    仕様:
      1. 全時刻観測された領域 (hits == n_time) かつ有限値 (np.isfinite) の点のみを抽出。
      2. 抽出点数 (n_valid) が 10 点未満の場合は status = "insufficient_points", sigma_0 = 0.0 とする。
      3. n_valid >= 10 の場合、中央値絶対偏差 (MAD) により基本雑音水準 sigma_0 を算出:
         - mad = median(|s - median(s)|)
         - mad > 0 の場合: sigma_0 = 1.4826 * mad, status = "ok"
         - mad == 0 の場合 (完全無雑音・完全平坦等): sigma_0 = 0.0, status = "zero_mad" とする。
           (輝線成分による std 誤認を防ぐため、MAD==0 を雑音推定不能として扱う)
      4. 各 sky チャンネルへのヒット数補正:
         sigma_d = sigma_0 * sqrt(n_time / hits_d)  (hits_d > 0)
         hits_d == 0 または NaN のチャンネルは sigma_d = np.nan。

    Returns
    -------
    sigma_d : (n_wide,) 各チャンネルの推定標準偏差 [K]
    sigma_0 : 全時刻観測領域での推定標準偏差 [K]
    status : 雑音推定状態 ("ok", "insufficient_points", "zero_mad")
    n_valid : 有効抽出点数
    """
    fully_observed = (hits == n_time) & np.isfinite(spectrum)
    valid_points = spectrum[fully_observed]
    n_valid = int(len(valid_points))

    if n_valid < 10:
        status = "insufficient_points"
        sigma_0 = 0.0
    else:
        med = float(np.median(valid_points))
        mad = float(np.median(np.abs(valid_points - med)))
        if mad > 0.0:
            sigma_0 = 1.4826 * mad
            status = "ok"
        else:
            sigma_0 = 0.0
            status = "zero_mad"

    sigma_d = np.full_like(spectrum, np.nan, dtype=float)
    valid_hits = hits > 0
    if sigma_0 > 0.0:
        sigma_d[valid_hits] = sigma_0 * np.sqrt(n_time / hits[valid_hits])
    else:
        sigma_d[valid_hits] = 0.0

    return sigma_d, sigma_0, status, n_valid


def make_noise_free_model(
    spectrum: np.ndarray,
    hits: np.ndarray,
    n_time: int,
    cutoff_sigma: float,
    allow_noise_free: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, str, int]:
    """sigma-cutoff 法による天体スペクトルモデルの作成

    Parameters
    ----------
    spectrum : (n_wide,) 復調・時間平均スペクトル [K]
    hits : (n_wide,) 観測ヒット数
    n_time : 時間サンプル数
    cutoff_sigma : 閾値 SNR (theta_cutoff)。論文の実証解析では 5.0 が採用された。
    allow_noise_free : 雑音推定失敗時 (status != 'ok') に例外を送出せず、
                       完全無雑音フォールバック (非ゼロ全成分採用) を許可するかどうか。
                       テスト専用オプション。

    Returns
    -------
    model : (n_wide,) 天体スペクトルモデル [K] (未観測・閾値以下は 0.0)
    sigma_d : (n_wide,) 推定雑音水準 [K]
    sigma_0 : 基準雑音水準 [K]
    status : 雑音推定状態 ("ok", "insufficient_points", "zero_mad")
    n_valid : 有効抽出点数
    """
    sigma_d, sigma_0, status, n_valid = estimate_noise_sigma(spectrum, hits, n_time)
    model = np.zeros_like(spectrum, dtype=float)

    if status != "ok":
        if not allow_noise_free:
            raise ValueError(
                f"雑音水準の推定に失敗しました (status='{status}', 有効点数={n_valid})。"
                "天体モデルの作成を中断します。雑音なしデータで実行する場合は allow_noise_free=True を指定してください。"
            )
        # 完全無雑音時のフォールバック (allow_noise_free=True の場合のみ許可)
        mask = (hits > 0) & np.isfinite(spectrum) & (np.abs(spectrum) > 0.0)
        model[mask] = spectrum[mask]
        return model, sigma_d, sigma_0, status, n_valid

    cutoff = cutoff_sigma * sigma_d
    mask = (hits > 0) & np.isfinite(spectrum) & (np.abs(spectrum) > cutoff)
    model[mask] = spectrum[mask]

    return model, sigma_d, sigma_0, status, n_valid
def run_single_iteration(
    r: int,
    x_tod: np.ndarray,
    a_prev_tod: np.ndarray,
    shifts_ch: np.ndarray,
    start0: int,
    n_wide: int,
    n_components: int,
    mean_mode: str,
    cutoff_sigma: float,
    allow_noise_free: bool = False,
) -> IterationResult:
    """r 回目の成分推定ステップを実行する

    Parameters
    ----------
    r : 反復回数 (1 または 2)
    x_tod : (n_time, n_channels) 元の観測 TOD X
    a_prev_tod : (n_time, n_channels) 前回推定した天体信号 TOD A_(r-1) (r=1 では 0)
    shifts_ch : (n_time,) シフト量 [ch]
    start0 : 広い周波数グリッド上の基準インデックス
    n_wide : 広い周波数グリッドのチャンネル数
    n_components : PCA 除去成分数 K
    mean_mode : "none" または "channel_mean"
    cutoff_sigma : sigma-cutoff 閾値

    Returns
    -------
    IterationResult
    """
    n_time, n_channels = x_tod.shape

    # 1. U_r = X - A_(r-1)
    u_tod = x_tod - a_prev_tod

    # 2. PCA による相関成分 C_r の推定
    pca_res = removal.pca_remove(u_tod, n_components=n_components, mean_mode=mean_mode)
    c_tod = pca_res.removed

    # 3. X - C_r を復調・時間平均し、天体スペクトルモデルを推定
    clean_signal = x_tod - c_tod
    restored, hits = modulation.demodulate(clean_signal, shifts_ch, start0, n_wide)
    model, sigma_d, _, noise_status, noise_n_valid = make_noise_free_model(
        restored, hits, n_time, cutoff_sigma, allow_noise_free=allow_noise_free
    )

    # 4. モデルを再変調して A_r を作成
    a_tod = modulation.modulate(model, shifts_ch, start0, n_channels)

    # 5. 残差 E_r = X - C_r - A_r
    e_tod = x_tod - c_tod - a_tod

    return IterationResult(
        r=r,
        u_tod=u_tod,
        c_tod=c_tod,
        a_tod=a_tod,
        e_tod=e_tod,
        restored_spectrum=restored,
        model_spectrum=model,
        noise_sigma=sigma_d,
        hits=hits,
        explained_ratio=pca_res.explained_ratio,
        noise_status=noise_status,
        noise_n_valid=noise_n_valid,
    )


def run_two_step_estimation(
    observed_tod: np.ndarray,
    shifts_ch: np.ndarray,
    start0: int,
    n_wide: int,
    n_components: int,
    mean_mode: str = "none",
    cutoff_sigma: float = 5.0,
    allow_noise_free: bool = False,
) -> TwoStepResult:
    """Taniguchi et al. (2020) Fig. 6 に対応する 2 回の成分推定を実行する

    入力は観測データ・シフト量・グリッド情報のみであり、正解データや真の雑音は一切含まない。

    Parameters
    ----------
    observed_tod : (n_time, n_channels) 観測 TOD X
    shifts_ch : (n_time,) 各時刻の変調シフト量 [ch]
    start0 : 広い周波数グリッド上の基準インデックス
    n_wide : 広い周波数グリッドのチャンネル数
    n_components : PCA 除去成分数 K
    mean_mode : "none" または "channel_mean"
    cutoff_sigma : sigma-cutoff 閾値 theta_cutoff (論文実証解析での典型条件: 5.0)

    Returns
    -------
    TwoStepResult
    """
    x_tod = np.asarray(observed_tod, dtype=float)
    a0_tod = np.zeros_like(x_tod)

    # 1 回目 (r=1): U_1 = X - A_0 = X
    iter1 = run_single_iteration(
        r=1,
        x_tod=x_tod,
        a_prev_tod=a0_tod,
        shifts_ch=shifts_ch,
        start0=start0,
        n_wide=n_wide,
        n_components=n_components,
        mean_mode=mean_mode,
        cutoff_sigma=cutoff_sigma,
        allow_noise_free=allow_noise_free,
    )

    # 2 回目 (r=2): U_2 = X - A_1
    iter2 = run_single_iteration(
        r=2,
        x_tod=x_tod,
        a_prev_tod=iter1.a_tod,
        shifts_ch=shifts_ch,
        start0=start0,
        n_wide=n_wide,
        n_components=n_components,
        mean_mode=mean_mode,
        cutoff_sigma=cutoff_sigma,
        allow_noise_free=allow_noise_free,
    )

    return TwoStepResult(x_tod=x_tod, iter1=iter1, iter2=iter2)