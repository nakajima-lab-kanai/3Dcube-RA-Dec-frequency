"""PCA による相関雑音の除去

sklearn の PCA は必ず各チャンネルの時間平均を引くので、平均の扱いを明示的に
選べるよう numpy の SVD で実装している。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PcaResult:
    cleaned: np.ndarray         # 除去後のデータ (n_time, n_channels) [K]
    removed: np.ndarray         # 除去した成分(平均スペクトルを含む) (n_time, n_channels) [K]
    mean_spectrum: np.ndarray   # 引いた平均スペクトル (n_channels,) [K]。mean_mode='none' なら 0
    components: np.ndarray      # 除去した主成分のスペクトル (n_components, n_channels)
    time_coeffs: np.ndarray     # 各主成分の時間係数 (n_time, n_components) [K]
    explained_ratio: np.ndarray  # 各主成分の寄与率(特異値の2乗の比)


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
