"""雑音成分の生成

各成分は (n_time, n_channels) の配列 [K] として個別に生成し、NoiseComponents に
名前付きで登録する。同じ形の配列を add() すれば成分を差し替え・追加できる。

- 空の周波数に固定された成分(大気):広い周波数グリッド上で形を定義し、
  天体信号と同じ窓で切り出す。atmosphere_noise() の timeseries_k に外部の時系列
  (将来は Gato による大気の時系列)を渡せば、時間変動だけ差し替えられる。
- チャンネル(IF)に固定された成分(将来の装置由来の雑音):窓をずらしても
  動かないので、(n_time, n_channels) の配列をそのまま add() する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .modulation import modulate


def radiometer_sigma_k(tsys_k, channel_width_hz, dt_s):
    """ラジオメーター方程式 ΔT = Tsys / sqrt(Δν_ch × Δt) [K]"""
    return tsys_k / np.sqrt(channel_width_hz * dt_s)


def thermal_noise(n_time, n_channels, tsys_k, channel_width_hz, dt_s, rng):
    """熱雑音。各時刻・各チャンネルで独立なガウス雑音 [K]"""
    sigma = radiometer_sigma_k(tsys_k, channel_width_hz, dt_s)
    return rng.normal(0.0, sigma, size=(n_time, n_channels))


def colored_noise_timeseries(n_time, dt_s, alpha, rms, rng):
    """パワースペクトルが f^(-alpha) に従う有色雑音の時系列

    白色雑音をフーリエ空間で f^(-alpha/2) 倍して作る。平均は 0、標準偏差は rms。
    """
    freqs = np.fft.rfftfreq(n_time, dt_s)
    spec = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    scale = np.zeros_like(freqs)
    scale[1:] = freqs[1:] ** (-alpha / 2.0)
    series = np.fft.irfft(spec * scale, n=n_time)
    std = series.std()
    if std == 0:
        return np.zeros(n_time)
    return series * (rms / std)


def atmosphere_shape(freq_ghz, center_freq_ghz, bandwidth_ghz, slope_frac):
    """大気成分の周波数方向の形(無次元)。定数 1 に緩やかな傾きを加えたもの

    空の周波数の関数として定義する。変調なしの帯域の両端で 1 ± slope_frac になる。
    """
    x = (np.asarray(freq_ghz) - center_freq_ghz) / (bandwidth_ghz / 2.0)
    return 1.0 + slope_frac * x


def atmosphere_noise(timeseries_k, shape_wide, shifts_ch, start0, n_channels, offset_k=0.0):
    """大気成分 [K]:(offset_k + timeseries_k[t]) × 形(空の周波数)

    timeseries_k : (n_time,) 全チャンネル共通の時間変動 [K]。外部の時系列で差し替え可
    shape_wide   : (n_wide,) 広い周波数グリッド上の形(atmosphere_shape の戻り値)
    """
    window_shape = modulate(shape_wide, shifts_ch, start0, n_channels)
    return (offset_k + np.asarray(timeseries_k))[:, None] * window_shape


@dataclass
class NoiseComponents:
    """雑音成分の入れ物。成分名 → (n_time, n_channels) の配列 [K]"""

    shape: tuple
    components: dict = field(default_factory=dict)

    def add(self, name, tod):
        tod = np.asarray(tod)
        if tod.shape != tuple(self.shape):
            raise ValueError(f"雑音成分 '{name}' の形が {tod.shape} です(期待値 {self.shape})")
        self.components[name] = tod

    def total(self):
        out = np.zeros(self.shape)
        for tod in self.components.values():
            out += tod
        return out
