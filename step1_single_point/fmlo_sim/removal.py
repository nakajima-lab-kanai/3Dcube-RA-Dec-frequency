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


# ---------------------------------------------------------------------------
# 反復 PCA
# ---------------------------------------------------------------------------

@dataclass
class IterativePcaResult:
    """反復 PCA の結果。X = removed + model_tod + (cleaned - model_tod) が成り立つ

    cleaned   : X - C_final(復調・グリッディングに使う)
    removed   : 最終回の相関成分 C_final
    model_tod : 最終回の輝線モデル A_final(再投影した時系列)
    """

    cleaned: list               # パスごとの (n_time, n_channels) [K]
    removed: list
    model_tod: list
    pca: list                   # 最終回のパスごとの PcaResult
    n_iter: int
    converged: bool
    rel_change: list            # 各回の |A_i - A_(i-1)|_F / |A_i|_F(1 回目は None)
    model_norm: list            # 各回の |A_i|_F [K]


def iterative_pca(x_list, n_components, mean_mode, model_fn, max_iter=20, tol=0.005):
    """輝線モデルを差し引きながら PCA をくり返す(Taniguchi et al. 2020 §3.2 の反復)

    x_list   : 観測時系列のリスト(スキャンのパスごと。固定点なら長さ 1)
    model_fn : [X_p - C_p] のリストを受け取り、輝線モデルの時系列 [A_p] のリストを返す関数
               (固定点なら 復調 → σ-cutoff → 再変調、スキャンなら グリッディング →
               σ-cutoff → 逆投影)
    手順:
      i = 1 : C^(1) = PCA(X)、A^(1) = model_fn(X - C^(1))
      i > 1 : C^(i) = PCA(X - A^(i-1))、A^(i) = model_fn(X - C^(i))
      |A^(i) - A^(i-1)|_F / |A^(i)|_F < tol(全パスまとめたノルム)か max_iter で終了。
      A が 0 のまま(輝線が検出されない)なら 2 回目で収束とみなす。
    """
    xs = [np.asarray(x, dtype=float) for x in x_list]
    a_prev = [np.zeros_like(x) for x in xs]
    rel_change, model_norm = [], []
    converged = False
    pcas = removed = a_new = None
    for it in range(1, int(max_iter) + 1):
        pcas = [pca_remove(x - a, n_components, mean_mode) for x, a in zip(xs, a_prev)]
        removed = [p.removed for p in pcas]
        a_new = [np.asarray(a, dtype=float) for a in model_fn([x - c for x, c in zip(xs, removed)])]
        norm_new = float(np.sqrt(sum(np.sum(a ** 2) for a in a_new)))
        model_norm.append(norm_new)
        if it == 1:
            rel_change.append(None)
        else:
            diff = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(a_new, a_prev))))
            rel = 0.0 if norm_new == 0 and diff == 0 else (np.inf if norm_new == 0 else diff / norm_new)
            rel_change.append(float(rel))
            if rel < tol:
                converged = True
                a_prev = a_new
                break
        a_prev = a_new
    return IterativePcaResult(
        cleaned=[x - c for x, c in zip(xs, removed)], removed=removed, model_tod=a_prev, pca=pcas,
        n_iter=len(model_norm), converged=converged, rel_change=rel_change, model_norm=model_norm,
    )
