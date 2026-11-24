"""FMLO 変調・復調

各時刻で分光計の観測窓をずらし、広い周波数グリッドから窓を切り出す。
np.roll のような循環シフトは使わない(端の回り込みを避けるため)。
シフト量の符号は「正 = 窓が高周波側へ動く」。
"""
from __future__ import annotations

from math import gcd

import numpy as np


def make_shift_pattern(pattern, n_time, half_width_ch, zigzag_step_ch=1, rng=None, cycle_jitter=False):
    """各時刻のシフト量 [ch](整数)を作る

    pattern:
      'zigzag' : 0 から始めて 1サンプルあたり zigzag_step_ch ずつ動き、
                 ±half_width_ch で折り返す三角波
      'random' : [-half_width_ch, +half_width_ch] の一様な整数乱数
    cycle_jitter (zigzag のみ):
      zigzag の 1 周期ごとに [-floor(step/2), +floor(step/2)] の一様な整数オフセットを
      足す。変調が厳密に周期的だと、その周期の時間変動を PCA が相関成分として拾いやすい
      ので、周期性を崩すためのもの。シフト量の上限は half_width_ch + floor(step/2) になる。
      乱数は rng から引く(cycle_jitter = False なら rng は使わない = 従来と同じ系列)。
    """
    a = int(half_width_ch)
    if cycle_jitter and pattern != "zigzag":
        raise ValueError("cycle_jitter は zigzag パターンだけで使えます")
    if a == 0:
        return np.zeros(n_time, dtype=int)
    if pattern == "zigzag":
        pos = np.arange(n_time) * int(zigzag_step_ch)
        shifts = a - np.abs((pos + a) % (4 * a) - 2 * a)
        if cycle_jitter:
            shifts = shifts + cycle_jitter_offsets(n_time, a, int(zigzag_step_ch), rng)
        return shifts
    if pattern == "random":
        if rng is None:
            raise ValueError("random パターンには rng が必要です")
        return rng.integers(-a, a + 1, size=n_time)
    raise ValueError(f"未知の変調パターンです: {pattern}")


def cycle_jitter_offsets(n_time, half_width_ch, zigzag_step_ch, rng):
    """zigzag の周期ごとに一定のランダムオフセット [ch] を返す (n_time,)

    周期は compute_zigzag_period_samples の最小反復周期。オフセットは
    [-floor(step/2), +floor(step/2)] の一様な整数。
    """
    j = int(zigzag_step_ch) // 2
    period, _ = compute_zigzag_period_samples(half_width_ch, zigzag_step_ch, 1.0)
    if j == 0 or period is None:
        return np.zeros(n_time, dtype=int)
    if rng is None:
        raise ValueError("cycle_jitter には rng が必要です")
    n_cycles = -(-n_time // period)
    offsets = rng.integers(-j, j + 1, size=n_cycles)
    return offsets[np.arange(n_time) // period]


def shifts_from_config(mod_cfg, n_time, rng):
    """ModulationConfig からシフト量を作る(固定点観測・スキャン共通)"""
    return make_shift_pattern(mod_cfg.pattern, n_time, mod_cfg.half_width_ch,
                              mod_cfg.zigzag_step_ch, rng, cycle_jitter=mod_cfg.cycle_jitter)


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


def demodulate_tod(tod, shifts_ch, start0, n_wide):
    """各時刻の観測窓を空の周波数(広いグリッド)に戻す(時間平均はしない)

    Returns
    -------
    demod : (n_time, n_wide) 復調した時系列 [K]。その時刻に観測していない周波数は NaN
    weight : (n_time, n_wide) 有効サンプルの重みマスク W^NaN(観測した = 1、NaN = 0)

    重み付き時間平均 sum(W * nan_to_num(demod)) / sum(W) は demodulate() の
    復元スペクトルと一致し、sum(W, axis=0) はヒット数になる。
    """
    tod = np.asarray(tod)
    n_time, n_channels = tod.shape
    starts = window_starts(shifts_ch, start0, n_channels, n_wide)
    idx = starts[:, None] + np.arange(n_channels)[None, :]
    rows = np.arange(n_time)[:, None]
    demod = np.full((n_time, n_wide), np.nan, dtype=np.result_type(tod.dtype, np.float32))
    weight = np.zeros((n_time, n_wide), dtype=np.uint8)
    demod[rows, idx] = tod
    weight[rows, idx] = 1
    return demod, weight


def compute_zigzag_period_samples(half_width_ch: int, zigzag_step_ch: int, dt_s: float):
    """zigzag 変調の出力系列の最小反復周期をサンプル数と秒で返す。

    make_shift_pattern の生成式 pos = t * step,
    shift = a - |(pos + a) % (4*a) - 2*a| に基づき、
    内部位相 pos の循環周期 4*a / gcd(4*a, step) を導出し、
    その1周期分の出力列から最小反復周期を求める。

    返り値:
      period_samples : 出力系列の最小反復周期 [サンプル] (定数列なら None)
      period_seconds : period_samples * dt_s [秒] (定数列なら None)

    半幅 0 の定数列、および折り畳みで定数列になるケースでは
    非自明な周期を持たないため (None, None) を返す。
    """
    a = int(half_width_ch)
    step = int(zigzag_step_ch)
    if a == 0:
        return None, None
    # 内部位相 pos = t * step の循環周期: 4*a が step の倍数になる最小の t
    internal_period = 4 * a // gcd(4 * a, step)
    # 1周期分の出力列を生成
    ts = np.arange(internal_period)
    pos = ts * step
    shifts = a - np.abs((pos + a) % (4 * a) - 2 * a)
    # internal_period の約数の中で最小の反復周期を探す
    for p in range(1, internal_period + 1):
        if internal_period % p == 0:
            if np.all(shifts == np.tile(shifts[:p], internal_period // p)):
                if p == 1:
                    # 定数列：非自明な周期なし
                    return None, None
                return p, p * float(dt_s)
    return internal_period, internal_period * float(dt_s)
