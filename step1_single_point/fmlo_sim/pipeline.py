"""段階1の処理の流れ(模擬観測の生成 → 雑音除去 → 復調 → 評価)"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cube as cube_mod
from . import evaluation, modulation, noise, removal
from .config import MEAN_MODES, Config, spawn_rngs


@dataclass
class Simulation:
    """模擬観測データ一式"""

    cfg: Config
    grid: cube_mod.FrequencyGrid
    cube: np.ndarray            # (n_ra, n_dec, n_wide) [K]
    ra_arcsec: np.ndarray
    dec_arcsec: np.ndarray
    pixel: tuple                # スペクトルを取り出したピクセル (i_ra, i_dec)
    truth_k: np.ndarray         # 正解スペクトル (n_wide,) [K]
    time_s: np.ndarray          # (n_time,) [s]
    shifts_ch: np.ndarray       # 各時刻のシフト量 (n_time,) [ch]
    signal_tod: np.ndarray      # 変調後の天体信号のみ (n_time, n_channels) [K]
    noise: noise.NoiseComponents
    observed: np.ndarray        # 天体信号 + 全雑音 (n_time, n_channels) [K]


@dataclass
class Reduction:
    """雑音除去 → 復調 → 評価の結果"""

    label: str
    n_components: int
    mean_mode: str
    pca: removal.PcaResult
    restored_k: np.ndarray      # 復元スペクトル (n_wide,) [K]
    hits: np.ndarray
    metrics: dict


def simulate(cfg: Config) -> Simulation:
    rngs = spawn_rngs(cfg.seed)
    grid = cube_mod.make_frequency_grid(cfg)
    n_time, n_ch = cfg.n_time, grid.n_channels
    dt = cfg.modulation.dt_s

    # 1. 正解キューブ
    cube, ra, dec = cube_mod.make_gaussian_cube(cfg, grid)

    # 2. 1点のスペクトル(既定は天体中心)
    p_ra = cfg.source.center_ra_arcsec if cfg.pointing.ra_arcsec is None else cfg.pointing.ra_arcsec
    p_dec = cfg.source.center_dec_arcsec if cfg.pointing.dec_arcsec is None else cfg.pointing.dec_arcsec
    truth, pixel = cube_mod.extract_spectrum(cube, ra, dec, p_ra, p_dec)

    # 3. FMLO 変調
    shifts = modulation.make_shift_pattern(
        cfg.modulation.pattern, n_time, cfg.modulation.half_width_ch,
        cfg.modulation.zigzag_step_ch, rngs["modulation"],
    )
    signal_tod = modulation.modulate(truth, shifts, grid.start0, n_ch)

    # 4. 雑音(成分ごとに生成して登録する)
    comps = noise.NoiseComponents((n_time, n_ch))
    if cfg.noise.thermal.enabled:
        comps.add("thermal", noise.thermal_noise(
            n_time, n_ch, cfg.noise.thermal.tsys_k,
            cfg.spectrometer.channel_width_mhz * 1e6, dt, rngs["thermal"],
        ))
    if cfg.noise.atmosphere.enabled:
        atm = cfg.noise.atmosphere
        series = noise.colored_noise_timeseries(n_time, dt, atm.alpha, atm.rms_k, rngs["atmosphere"])
        shape = noise.atmosphere_shape(
            grid.freq_ghz, grid.center_freq_ghz, grid.bandwidth_ghz, atm.slope_frac
        )
        comps.add("atmosphere", noise.atmosphere_noise(
            series, shape, shifts, grid.start0, n_ch, atm.offset_k
        ))

    return Simulation(
        cfg=cfg, grid=grid, cube=cube, ra_arcsec=ra, dec_arcsec=dec, pixel=pixel,
        truth_k=truth, time_s=np.arange(n_time) * dt, shifts_ch=shifts,
        signal_tod=signal_tod, noise=comps, observed=signal_tod + comps.total(),
    )


def reduce(sim: Simulation, n_components: int, mean_mode: str, label=None) -> Reduction:
    """PCA で上位成分を除去し、復調・時間平均して評価する"""
    cfg, grid = sim.cfg, sim.grid
    pca = removal.pca_remove(sim.observed, n_components, mean_mode)
    restored, hits = modulation.demodulate(pca.cleaned, sim.shifts_ch, grid.start0, grid.n_wide)
    metrics = evaluation.evaluate(
        restored, sim.truth_k, hits, cfg.n_time, grid.freq_ghz,
        cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
    )
    return Reduction(label or mean_mode, n_components, mean_mode, pca, restored, hits, metrics)


def reduce_all_modes(sim: Simulation) -> dict:
    """同じ観測データに対して、平均の扱いを変えた結果を並べて返す

    'no_pca' は PCA も平均除去も行わず復調だけした参照用の結果。
    """
    k = sim.cfg.removal.n_components
    results = {mode: reduce(sim, k, mode) for mode in MEAN_MODES}
    results["no_pca"] = reduce(sim, 0, "none", label="no_pca")
    return results


def expected_thermal_rms_k(cfg: Config) -> float:
    """熱雑音だけを時間平均したときに期待される RMS [K]"""
    if not cfg.noise.thermal.enabled:
        return 0.0
    sigma = noise.radiometer_sigma_k(
        cfg.noise.thermal.tsys_k, cfg.spectrometer.channel_width_mhz * 1e6, cfg.modulation.dt_s
    )
    return float(sigma / np.sqrt(cfg.n_time))
