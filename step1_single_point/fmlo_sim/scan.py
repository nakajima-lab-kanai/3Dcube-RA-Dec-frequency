"""1 ビーム OTF スキャン + FMLO の模擬観測(段階 2-B)

サンプリング間隔 Δt(= 変調の間隔 modulation.dt_s)ごとに指向位置 (RA(t), Dec(t)) を
動かしながら、同時に LO を変調する。各時刻の位置で正解キューブを空間方向に
バイリニア補間してスペクトル(広い周波数グリッド上)を取り出し、その時刻のシフト量の
観測窓を切り出して「時間 × 分光器チャンネル」の時系列を作る。

スキャンの形(scan.mode):
  single_point : 1 点を見続ける(段階 1 と同じ。ただし取り出しはバイリニア補間)
  x_scan       : RA 方向に往復するラスター(行ごとに Dec を row_spacing ずつ進める)
  y_scan       : Dec 方向に往復するラスター
  basket_weave : x_scan と y_scan の 2 パス(別々の時刻に観測した独立なデータ)

簡略化していること(README 参照):
  - 行の折り返しは瞬時(加減速・ランプ区間・OFF 点なし)。行と行のあいだも連続にサンプリングする
  - 大気・装置の雑音は時間だけの関数(指向位置・仰角に依存しない)
  - 各パスの雑音と変調は独立(config.spawn_rngs(seed, pass_index))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cube as cube_mod
from . import modulation, noise, observation
from .config import Config, spawn_rngs


@dataclass
class ScanPass:
    """1 回分のスキャンの軌跡"""

    name: str                 # 'single_point' / 'x_scan' / 'y_scan'
    time_s: np.ndarray        # (n_time,) [s]
    ra_arcsec: np.ndarray     # (n_time,) [arcsec]
    dec_arcsec: np.ndarray    # (n_time,) [arcsec]
    row: np.ndarray           # (n_time,) 行番号(single_point は 0)

    @property
    def n_time(self) -> int:
        return self.time_s.size


def raster_pass(direction, map_size_arcsec, speed_arcsec_s, row_spacing_arcsec, dt_s,
                center_ra_arcsec=0.0, center_dec_arcsec=0.0) -> ScanPass:
    """往復ラスタースキャンの軌跡

    direction = 'x' なら RA 方向に走査し、行ごとに Dec を進める('y' はその逆)。
    1 行のサンプル数は round(L / (v Δt)) + 1(両端を含む)、行数は round(L / 行間隔) + 1。
    奇数行は逆向きに走査する(折り返しは瞬時)。
    """
    if direction not in ("x", "y"):
        raise ValueError(f"direction は 'x' か 'y' です: {direction}")
    half = map_size_arcsec / 2.0
    step = speed_arcsec_s * dt_s
    n_along = int(round(map_size_arcsec / step)) + 1
    n_rows = int(round(map_size_arcsec / row_spacing_arcsec)) + 1
    along = -half + np.arange(n_along) * step
    across = np.minimum(-half + np.arange(n_rows) * row_spacing_arcsec, half)
    a_list, c_list, r_list = [], [], []
    for k in range(n_rows):
        a_list.append(along if k % 2 == 0 else along[::-1])
        c_list.append(np.full(n_along, across[k]))
        r_list.append(np.full(n_along, k))
    a, c, r = np.concatenate(a_list), np.concatenate(c_list), np.concatenate(r_list)
    ra, dec = (a, c) if direction == "x" else (c, a)
    n = a.size
    return ScanPass(f"{direction}_scan", np.arange(n) * dt_s, ra + center_ra_arcsec,
                    dec + center_dec_arcsec, r)


def make_passes(cfg: Config) -> list:
    """scan.mode に従ってスキャンのパスのリストを作る"""
    sc, dt = cfg.scan, cfg.modulation.dt_s
    if sc.mode == "single_point":
        n = cfg.n_time
        return [ScanPass("single_point", np.arange(n) * dt, np.full(n, sc.center_ra_arcsec),
                         np.full(n, sc.center_dec_arcsec), np.zeros(n, dtype=int))]
    args = (sc.map_size_arcsec, sc.scan_speed_arcsec_s, sc.row_spacing_arcsec, dt,
            sc.center_ra_arcsec, sc.center_dec_arcsec)
    if sc.mode == "x_scan":
        return [raster_pass("x", *args)]
    if sc.mode == "y_scan":
        return [raster_pass("y", *args)]
    if sc.mode == "basket_weave":
        return [raster_pass("x", *args), raster_pass("y", *args)]
    raise ValueError(f"未知の scan.mode です: {sc.mode}")


def sampling_summary(cfg: Config, passes) -> dict:
    """サンプリングの条件(ビームに対するサンプル間隔・行間隔など)の記録"""
    sc, dt = cfg.scan, cfg.modulation.dt_s
    beam = cfg.telescope.beam_fwhm_arcsec
    return {
        "mode": sc.mode,
        "passes": [{"name": p.name, "n_time": p.n_time, "duration_s": p.n_time * dt,
                    "n_rows": int(p.row.max()) + 1} for p in passes],
        "sample_spacing_arcsec": sc.scan_speed_arcsec_s * dt,
        "row_spacing_arcsec": sc.row_spacing_arcsec,
        "beam_fwhm_arcsec": beam,
        # ナイキスト間隔は λ/(2D) ≈ FWHM / 2.4。行間隔・サンプル間隔がこれを超えると空間的に粗い
        "nyquist_spacing_arcsec_approx": beam / 2.4,
    }


@dataclass
class ScanObservation:
    """1 パス分の模擬観測。signal_tod / noise は正解情報(評価専用)"""

    scan_pass: ScanPass
    grid: cube_mod.FrequencyGrid
    shifts_ch: np.ndarray       # (n_time,) [ch]
    signal_tod: np.ndarray      # 天体信号のみ (n_time, n_channels) [K]
    noise: noise.NoiseComponents
    observed: np.ndarray        # 天体信号 + 全雑音 (n_time, n_channels) [K]


def observe_pass(sky: observation.SkyCube, cfg: Config, scan_pass: ScanPass, pass_index: int) -> ScanObservation:
    """キューブ上を scan_pass の軌跡で動きながら FMLO 観測した時系列を作る(入力は変更しない)

    乱数は spawn_rngs(seed, pass_index) の成分別生成器を使う(パスごとに独立)。
    外部大気(model = external)の .npz は、配列 'atmosphere_<パス名>' があればそれを使う。
    """
    observation.validate_sky_cube(sky)
    grid = observation.match_frequency_grid(sky.freq_ghz, cfg.spectrometer)
    rngs = spawn_rngs(cfg.seed, pass_index)
    n_time, n_ch = scan_pass.n_time, grid.n_channels

    shifts = modulation.shifts_from_config(cfg.modulation, n_time, rngs["modulation"])
    observation.check_modulation_band(grid, shifts)

    # 各時刻の位置のスペクトル(バイリニア補間)→ その時刻の観測窓を切り出す
    wide = cube_mod.sample_spectra(sky.data, sky.ra_arcsec, sky.dec_arcsec,
                                   scan_pass.ra_arcsec, scan_pass.dec_arcsec)
    signal_tod = modulation.modulate(wide, shifts, grid.start0, n_ch)
    del wide

    comps = noise.make_noise_components(cfg.noise, cfg.spectrometer, grid, shifts, cfg.modulation.dt_s,
                                        rngs, external_key=f"atmosphere_{scan_pass.name}")
    return ScanObservation(scan_pass, grid, shifts, signal_tod, comps, signal_tod + comps.total())
