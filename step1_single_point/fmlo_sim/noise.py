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
from pathlib import Path

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


def laminar_noise_timeseries(n_time, dt_s, rms, rng=None):
    """層流による大気雑音の時系列（穏やかな低周波変動・ドリフト）

    乱流のような急激な高周波の揺らぎを持たず、観測時間全体にわたって滑らかに変化する
    基底変動（線形ドリフト＋緩やかな長周期成分）を生成する。平均は 0、標準偏差は rms。
    """
    t = np.linspace(-1.0, 1.0, n_time)
    if rng is not None:
        c1 = rng.normal(0.0, 1.0)
        c2 = rng.normal(0.0, 0.5)
        phase = rng.uniform(0.0, 2 * np.pi)
        series = c1 * t + c2 * (t**2 - 1.0 / 3.0) + 0.3 * np.cos(np.pi * t + phase)
    else:
        series = t + 0.5 * (t**2 - 1.0 / 3.0)
    
    series = series - series.mean()
    std = series.std()
    if std == 0:
        return np.zeros(n_time)
    return series * (rms / std)


def random_walk_timeseries(n_time, rms, rng):
    """正規乱数の累積和(ランダムウォーク)による緩やかなベースライン変動

    パワースペクトルはおおむね f^-2。平均 0、標準偏差 rms に規格化する。
    """
    series = np.cumsum(rng.normal(size=n_time))
    series = series - series.mean()
    std = series.std()
    if std == 0:
        return np.zeros(n_time)
    return series * (rms / std)


def load_external_atmosphere(path, n_time, n_wide, freq_wide_ghz=None, key=None):
    """外部シミュレータ(GATO など)が出力した大気の配列を読み込む

    対応する形:
      (n_time,)         : 全チャンネル共通の時間変動 a(t) [K]。(offset_k + a(t)) × s(ν) に使う
      (n_time, n_wide)  : 空の周波数(広いグリッド)ごとの大気の明るさ [K]。
                          天体と同じ観測窓で切り出す。offset_k と slope_frac は使わない
    .npz の場合は key(既定 'atmosphere')の配列を読む。'freq_ghz' があれば
    広いグリッドの周波数と一致するか確かめる。形・長さが合わなければエラー
    (補間・切り詰めはしない)。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"外部大気ファイルが見つかりません: {path}")
    freq = None
    if path.suffix == ".npz":
        with np.load(path) as f:
            name = key if key is not None and key in f.files else "atmosphere"
            if name not in f.files:
                raise ValueError(f"{path} に配列 '{name}' がありません(含まれる配列: {f.files})")
            arr = np.asarray(f[name], dtype=float)
            if "freq_ghz" in f.files:
                freq = np.asarray(f["freq_ghz"], dtype=float)
    elif path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
    else:
        raise ValueError(f"外部大気ファイルは .npy か .npz にしてください: {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} に有限でない値が含まれています")
    if arr.shape == (n_time,):
        return arr
    if arr.shape == (n_time, n_wide):
        if freq is not None and freq_wide_ghz is not None:
            if freq.shape != (n_wide,) or not np.allclose(freq, freq_wide_ghz, rtol=0, atol=1e-9):
                raise ValueError(f"{path} の freq_ghz が広い周波数グリッドと一致しません")
        return arr
    raise ValueError(
        f"外部大気の配列の形 {arr.shape} に対応していません。"
        f"(n_time,) = ({n_time},) か (n_time, n_wide) = ({n_time}, {n_wide}) にしてください"
    )


def instrument_noise(n_time, n_channels, dt_s, alpha, rms_k, rng):
    """装置由来の 1/f^alpha 雑音(全チャンネル共通の利得変動) [K]

    チャンネル(IF)に固定された成分なので、変調の窓では切り出さない。
    """
    series = colored_noise_timeseries(n_time, dt_s, alpha, rms_k, rng)
    return np.repeat(series[:, None], n_channels, axis=1)


def make_noise_components(noise_cfg, spectrometer, grid, shifts_ch, dt_s, rngs, external_key=None):
    """設定に従って雑音成分を作り、NoiseComponents に登録して返す

    固定点観測(observation.observe)とスキャン(scan.observe_pass)で共用する。
    乱数は rngs の成分別生成器だけを使う(成分ごとに独立)。
    """
    n_time, n_ch = len(shifts_ch), grid.n_channels
    comps = NoiseComponents((n_time, n_ch))
    if noise_cfg.thermal.enabled:
        comps.add("thermal", thermal_noise(
            n_time, n_ch, noise_cfg.thermal.tsys_k,
            spectrometer.channel_width_mhz * 1e6, dt_s, rngs["thermal"],
        ))
    atm = noise_cfg.atmosphere
    if atm.enabled:
        shape = atmosphere_shape(grid.freq_ghz, grid.center_freq_ghz, grid.bandwidth_ghz, atm.slope_frac)
        if atm.model in ("colored", "turbulent"):
            series = colored_noise_timeseries(n_time, dt_s, atm.alpha, atm.rms_k, rngs["atmosphere"])
        elif atm.model == "random_walk":
            series = random_walk_timeseries(n_time, atm.rms_k, rngs["atmosphere"])
        elif atm.model == "external":
            series = load_external_atmosphere(atm.external_path, n_time, grid.n_wide,
                                              grid.freq_ghz, key=external_key)
        else:
            series = laminar_noise_timeseries(n_time, dt_s, atm.rms_k, rngs["atmosphere"])
        if series.ndim == 2:
            # 空の周波数ごとの外部配列:天体と同じ窓で切り出す
            comps.add("atmosphere", modulate(series, shifts_ch, grid.start0, n_ch))
        else:
            comps.add("atmosphere", atmosphere_noise(series, shape, shifts_ch, grid.start0, n_ch, atm.offset_k))
    inst = noise_cfg.instrument
    if inst.enabled:
        comps.add("instrument", instrument_noise(n_time, n_ch, dt_s, inst.alpha, inst.rms_k, rngs["instrument"]))
    return comps


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
