"""FMLO 変調・復調

各時刻で分光計の観測窓をずらし、広い周波数グリッドから窓を切り出す。
np.roll のような循環シフトは使わない(端の回り込みを避けるため)。
シフト量の符号は「正 = 窓が高周波側へ動く」。
"""
from __future__ import annotations

import numpy as np


def make_shift_pattern(pattern, n_time, half_width_ch, zigzag_step_ch=1, rng=None):
    """各時刻のシフト量 [ch](整数)を作る

    pattern:
      'zigzag' : 0 から始めて 1サンプルあたり zigzag_step_ch ずつ動き、
                 ±half_width_ch で折り返す三角波
      'random' : [-half_width_ch, +half_width_ch] の一様な整数乱数
    """
    a = int(half_width_ch)
    if a == 0:
        return np.zeros(n_time, dtype=int)
    if pattern == "zigzag":
        pos = np.arange(n_time) * int(zigzag_step_ch)
        return a - np.abs((pos + a) % (4 * a) - 2 * a)
    if pattern == "random":
        if rng is None:
            raise ValueError("random パターンには rng が必要です")
        return rng.integers(-a, a + 1, size=n_time)
    raise ValueError(f"未知の変調パターンです: {pattern}")


def window_starts(shifts_ch, start0, n_channels, n_wide):
    """各時刻の窓の先頭インデックス。広いグリッドからはみ出す場合はエラー"""
    starts = start0 + np.asarray(shifts_ch, dtype=int)
    if starts.min() < 0 or starts.max() + n_channels > n_wide:
        raise ValueError(
            "観測窓が広い周波数グリッドの範囲を超えています。"
            "キューブの周波数範囲を広げるか、変調幅を小さくしてください"
        )
    return starts


def modulate(wide, shifts_ch, start0, n_channels):
    """広いグリッド上のスペクトルから各時刻の観測窓を切り出す

    wide : (n_wide,) なら時間変化しないスペクトル、
           (n_time, n_wide) なら時刻ごとのスペクトル(段階2以降のスキャン用)
    Returns : (n_time, n_channels) の配列
    """
    wide = np.asarray(wide)
    starts = window_starts(shifts_ch, start0, n_channels, wide.shape[-1])
    idx = starts[:, None] + np.arange(n_channels)[None, :]
    if wide.ndim == 1:
        return wide[idx]
    return np.take_along_axis(wide, idx, axis=1)


def demodulate(tod, shifts_ch, start0, n_wide):
    """各時刻の観測窓を元の周波数に戻し、時間方向に平均する

    Returns
    -------
    spectrum : (n_wide,) 時間平均スペクトル。一度も観測されなかった周波数は NaN
    hits : (n_wide,) 各周波数が観測された回数
    """
    n_time, n_channels = tod.shape
    starts = window_starts(shifts_ch, start0, n_channels, n_wide)
    total = np.zeros(n_wide)
    hits = np.zeros(n_wide, dtype=int)
    for t in range(n_time):
        total[starts[t]:starts[t] + n_channels] += tod[t]
        hits[starts[t]:starts[t] + n_channels] += 1
    spectrum = np.full(n_wide, np.nan)
    np.divide(total, hits, out=spectrum, where=hits > 0)
    return spectrum, hits
