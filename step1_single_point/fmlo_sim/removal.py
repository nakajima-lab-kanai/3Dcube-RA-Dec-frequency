"""PCA および FA+ICA による相関雑音の除去

- PCA:
  sklearn の PCA は必ず各チャンネルの時間平均を引くので、平均の扱いを明示的に
  選べるよう numpy の SVD で実装している。
- FA+ICA:
  FactorAnalysis (因子分析) で共通因子空間を推定し、FastICA (独立成分分析) で
  独立成分へ回転して、選択した成分のみを時間×チャンネル空間へ再構成して除去する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.decomposition import FactorAnalysis, FastICA


@dataclass
class PcaResult:
    cleaned: np.ndarray         # 除去後のデータ (n_time, n_channels) [K]
    removed: np.ndarray         # 除去した成分(平均スペクトルを含む) (n_time, n_channels) [K]
    mean_spectrum: np.ndarray   # 引いた平均スペクトル (n_channels,) [K]。mean_mode='none' なら 0
    components: np.ndarray      # 除去した主成分のスペクトル (n_components, n_channels)
    time_coeffs: np.ndarray     # 各主成分の時間係数 (n_time, n_components) [K]
    explained_ratio: np.ndarray  # 各主成分の寄与率(特異値の2乗の比)


@dataclass
class FaIcaResult:
    cleaned: np.ndarray              # 除去後のデータ (n_time, n_channels) [K]
    removed: np.ndarray              # 観測データから差し引いた全成分 (n_time, n_channels) [K]
    mean_spectrum: np.ndarray        # 差し引いた平均スペクトル (n_channels,) [K]
    n_factors: int                   # 推定した共通因子数 K
    remove_indices: list[int]        # 除去対象として選択した成分のインデックスリスト J
    mean_mode: str                   # 平均処理モード ('channel_mean' または 'none')
    # FA 固有の診断量
    fa_loadings: np.ndarray          # 因子負荷行列 L (n_factors, n_channels)
    fa_scores: np.ndarray            # 因子得点行列 Z (n_time, n_factors)
    fa_noise_variance: np.ndarray    # 独自分散 psi (n_channels,)
    fa_n_iter: int                   # FA 反復回数
    # ICA 固有の診断量
    ica_components: np.ndarray       # 独立成分 S (n_time, n_factors) [時間波形]
    ica_mixing: np.ndarray           # 混合行列 A (n_factors, n_factors)
    effective_profiles: np.ndarray   # 有効スペクトルプロファイル M = A.T @ L (n_factors, n_channels)
    ica_n_iter: int                  # ICA 反復回数


def pca_remove(tod, n_components, mean_mode="none"):
    """時間×チャンネルのデータから上位 n_components 個の主成分を除去する

    mean_mode:
      'none'         : 平均を引かずに SVD する(原点まわりの PCA)
      'channel_mean' : 各チャンネルの時間平均を引いてから SVD する(sklearn の PCA と同じ)。
                       引いた平均は戻さない(X - inverse_transform(transform(X)) と同じ結果)
    """
    tod = np.asarray(tod, dtype=float)
    n_channels = tod.shape[1]
    if mean_mode == "channel_mean":
        mean = tod.mean(axis=0)
    elif mean_mode == "none":
        mean = np.zeros(n_channels)
    else:
        raise ValueError(f"未知の mean_mode です: {mean_mode}")
    centered = tod - mean

    k = int(n_components)
    if k == 0:
        low_rank = np.zeros_like(centered)
        components = np.zeros((0, n_channels))
        coeffs = np.zeros((tod.shape[0], 0))
        ratio = np.zeros(0)
    else:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coeffs = u[:, :k] * s[:k]
        components = vt[:k]
        low_rank = coeffs @ components
        ratio = s[:k] ** 2 / np.sum(s ** 2)

    return PcaResult(
        cleaned=centered - low_rank,
        removed=low_rank + mean,
        mean_spectrum=mean,
        components=components,
        time_coeffs=coeffs,
        explained_ratio=ratio,
    )


def fa_ica_remove(
    tod: np.ndarray,
    n_factors: int,
    remove_indices: Optional[List[int]] = None,
    mean_mode: str = "channel_mean",
    random_state: int = 42,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> FaIcaResult:
    """時間×チャンネルのデータから FA + ICA により特定成分を除去する

    Parameters
    ----------
    tod : (n_time, n_channels) の観測データ配列 [K]
    n_factors : 推定する FA 共通因子数 (>= 1)
    remove_indices : 除去対象の ICA 成分インデックス (0-based)。None の場合は全成分 [0, ..., n_factors-1]
    mean_mode :
      'channel_mean' : 各チャンネルの時間平均を引いてから FA+ICA を適用し、引いた平均は戻さない
      'none'         : 外部で平均を引かず、時間変動成分のみを差し引く (平均は保持される)
    random_state : ICA および FA の初期化シード
    max_iter : 最大反復回数
    tol : 収束許容値
    """
    tod = np.asarray(tod, dtype=float)
    n_time, n_channels = tod.shape

    if mean_mode == "channel_mean":
        mean = tod.mean(axis=0)
    elif mean_mode == "none":
        mean = np.zeros(n_channels)
    else:
        raise ValueError(f"未知の mean_mode です: {mean_mode}")
    centered = tod - mean

    k = int(n_factors)
    if k < 1:
        raise ValueError(f"n_factors は 1 以上である必要があります: {n_factors}")

    if remove_indices is None:
        j_list = list(range(k))
    else:
        j_list = [int(i) for i in remove_indices]
        for idx in j_list:
            if not 0 <= idx < k:
                raise ValueError(f"remove_indices のインデックス {idx} が範囲外です (0 <= idx < {k})")

    # 1. FA (因子分析)
    fa = FactorAnalysis(
        n_components=k,
        svd_method="randomized",
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
    )
    z = fa.fit_transform(centered)   # (n_time, k)
    loadings = fa.components_        # (k, n_channels)
    psi = fa.noise_variance_         # (n_channels,)

    # 2. FastICA (独立成分分析)
    ica = FastICA(
        n_components=k,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        whiten="unit-variance",
    )
    s = ica.fit_transform(z)         # (n_time, k)
    mixing = ica.mixing_             # (k, k)
    profiles = mixing.T @ loadings   # (k, n_channels)

    # 3. 選択成分の再構成
    if len(j_list) == 0:
        fluc_removed = np.zeros_like(centered)
    else:
        fluc_removed = s[:, j_list] @ profiles[j_list, :]

    # 4. 平均の扱いと出力の組み立て
    if mean_mode == "channel_mean":
        cleaned = centered - fluc_removed
        removed = fluc_removed + mean
        mean_spectrum = mean
    else:  # "none"
        cleaned = tod - fluc_removed
        removed = fluc_removed
        mean_spectrum = np.zeros(n_channels)

    return FaIcaResult(
        cleaned=cleaned,
        removed=removed,
        mean_spectrum=mean_spectrum,
        n_factors=k,
        remove_indices=j_list,
        mean_mode=mean_mode,
        fa_loadings=loadings,
        fa_scores=z,
        fa_noise_variance=psi,
        fa_n_iter=int(fa.n_iter_),
        ica_components=s,
        ica_mixing=mixing,
        effective_profiles=profiles,
        ica_n_iter=int(ica.n_iter_),
    )
