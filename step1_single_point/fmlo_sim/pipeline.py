"""段階1の処理の流れ(模擬観測の生成 → 雑音除去 → 復調 → 評価)"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import cube as cube_mod
from . import evaluation, iterative, modulation, noise, observation, removal
from .config import MEAN_MODES, Config


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
    """内部生成のガウシアンキューブを固定点観測する(既存 API の互換ラッパー)

    キューブ生成(cube.make_gaussian_cube)と観測処理(observation.observe)は分離してあり、
    外部入力キューブも同じ observe() を通る。
    """
    sky = make_internal_sky_cube(cfg)
    obs = observation.observe(sky, internal_observing_setup(cfg))
    return simulation_from_observation(cfg, sky, obs)


def make_internal_sky_cube(cfg: Config) -> observation.SkyCube:
    """設定からガウシアンの正解キューブを作り、SkyCube として返す"""
    grid = cube_mod.make_frequency_grid(cfg)
    cube, ra, dec = cube_mod.make_gaussian_cube(cfg, grid)
    identity = {
        "kind": "internal_gaussian",
        "generator": "fmlo_sim.cube.make_gaussian_cube",
        "source": dataclasses.asdict(cfg.source),
        "cube": dataclasses.asdict(cfg.cube),
    }
    if cfg.telescope.beam_convolve:
        # 既定(beam_convolve = false)では畳み込まず、identity も従来どおり
        cube = cube_mod.convolve_beam(cube, cfg.cube.pixel_arcsec, cfg.telescope.beam_fwhm_arcsec)
        identity["beam_fwhm_arcsec"] = cfg.telescope.beam_fwhm_arcsec
    return observation.SkyCube(cube, ra, dec, grid.freq_ghz, identity=identity)


def internal_observing_setup(cfg: Config) -> observation.ObservingSetup:
    """Config から観測設定を取り出す。pointing が null なら天体中心(既存仕様)"""
    p_ra = cfg.source.center_ra_arcsec if cfg.pointing.ra_arcsec is None else cfg.pointing.ra_arcsec
    p_dec = cfg.source.center_dec_arcsec if cfg.pointing.dec_arcsec is None else cfg.pointing.dec_arcsec
    return observation.ObservingSetup.from_config(cfg, p_ra, p_dec)


def simulation_from_observation(cfg: Config, sky: observation.SkyCube,
                                obs: observation.Observation) -> Simulation:
    """observe() の結果を既存の Simulation に詰め替える"""
    return Simulation(
        cfg=cfg, grid=obs.grid, cube=sky.data, ra_arcsec=sky.ra_arcsec, dec_arcsec=sky.dec_arcsec,
        pixel=obs.pixel, truth_k=obs.truth_k, time_s=obs.time_s, shifts_ch=obs.shifts_ch,
        signal_tod=obs.signal_tod, noise=obs.noise, observed=obs.observed,
    )


@dataclass
class BlindReduction:
    """正解を使わない解析結果(PCA 除去 → 復調・時間平均)"""

    label: str
    n_components: int
    mean_mode: str
    pca: removal.PcaResult
    restored_k: np.ndarray      # 復元スペクトル (n_wide,) [K]。未観測の周波数は NaN
    hits: np.ndarray


def reduce_observed(observed, shifts_ch, start0, n_wide, n_components: int, mean_mode: str,
                    label=None) -> BlindReduction:
    """観測時系列・シフト量・格子情報だけから復元スペクトルを作る(正解は受け取らない)"""
    pca = removal.pca_remove(observed, n_components, mean_mode)
    restored, hits = modulation.demodulate(pca.cleaned, shifts_ch, start0, n_wide)
    return BlindReduction(label or mean_mode, n_components, mean_mode, pca, restored, hits)


def reduce_observed_all_modes(observed, shifts_ch, start0, n_wide, n_components: int) -> dict:
    """reduce_all_modes と同じ 3 通り(none / channel_mean / no_pca)を正解なしで実行する"""
    results = {
        mode: reduce_observed(observed, shifts_ch, start0, n_wide, n_components, mode)
        for mode in MEAN_MODES
    }
    results["no_pca"] = reduce_observed(observed, shifts_ch, start0, n_wide, 0, "none", label="no_pca")
    return results


def reduce(sim: Simulation, n_components: int, mean_mode: str, label=None) -> Reduction:
    """PCA で上位成分を除去し、復調・時間平均して評価する"""
    cfg, grid = sim.cfg, sim.grid
    blind = reduce_observed(sim.observed, sim.shifts_ch, grid.start0, grid.n_wide,
                            n_components, mean_mode, label)
    metrics = evaluation.evaluate(
        blind.restored_k, sim.truth_k, blind.hits, cfg.n_time, grid.freq_ghz,
        cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
    )
    return Reduction(blind.label, n_components, mean_mode, blind.pca, blind.restored_k, blind.hits, metrics)


def reduce_all_modes(sim: Simulation) -> dict:
    """同じ観測データに対して、平均の扱いを変えた結果を並べて返す

    'no_pca' は PCA も平均除去も行わず復調だけした参照用の結果。
    """
    k = sim.cfg.removal.n_components
    results = {mode: reduce(sim, k, mode) for mode in MEAN_MODES}
    results["no_pca"] = reduce(sim, 0, "none", label="no_pca")
    return results


@dataclass
class IterativeReduction(Reduction):
    """反復 PCA の結果(Reduction に反復の記録を足したもの)"""

    iterative: removal.IterativePcaResult = None


def point_model_fn(shifts_ch, start0, n_wide, n_channels, n_time, cutoff_sigma):
    """固定点観測の輝線モデル:復調・時間平均 → σ-cutoff → 再変調"""

    def model_fn(cleaned_list):
        out = []
        for cleaned in cleaned_list:
            spec, hits = modulation.demodulate(cleaned, shifts_ch, start0, n_wide)
            model, *_ = iterative.make_noise_free_model(spec, hits, n_time, cutoff_sigma)
            out.append(modulation.modulate(model, shifts_ch, start0, n_channels))
        return out

    return model_fn


def reduce_observed_iterative(observed, shifts_ch, start0, n_wide, n_components, mean_mode,
                              max_iter, tol, cutoff_sigma, label="iterative_pca"):
    """反復 PCA → 復調(正解は使わない)。BlindReduction と反復の記録を返す"""
    n_time, n_ch = np.shape(observed)
    res = removal.iterative_pca(
        [observed], n_components, mean_mode,
        point_model_fn(shifts_ch, start0, n_wide, n_ch, n_time, cutoff_sigma), max_iter, tol,
    )
    restored, hits = modulation.demodulate(res.cleaned[0], shifts_ch, start0, n_wide)
    return BlindReduction(label, n_components, mean_mode, res.pca[0], restored, hits), res


def reduce_iterative(sim: Simulation, n_components=None, mean_mode=None,
                     label="iterative_pca") -> IterativeReduction:
    """removal の設定(max_iter, tol, cutoff_sigma)で反復 PCA を行い、評価する"""
    cfg, grid = sim.cfg, sim.grid
    k = cfg.removal.n_components if n_components is None else n_components
    mode = cfg.removal.mean_mode if mean_mode is None else mean_mode
    blind, res = reduce_observed_iterative(
        sim.observed, sim.shifts_ch, grid.start0, grid.n_wide, k, mode,
        cfg.removal.max_iter, cfg.removal.tol, cfg.removal.cutoff_sigma, label,
    )
    metrics = evaluation.evaluate(
        blind.restored_k, sim.truth_k, blind.hits, cfg.n_time, grid.freq_ghz,
        cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
    )
    # 輝線モデルが 0 のまま(輝線が 1 チャンネルも検出されない)だと「収束」と判定されるが、
    # 反復は何もしていない。final_model_norm_k = 0 で見分ける
    metrics.update({"n_iter": res.n_iter, "converged": res.converged,
                    "final_rel_change": res.rel_change[-1], "final_model_norm_k": res.model_norm[-1]})
    return IterativeReduction(blind.label, k, mode, blind.pca, blind.restored_k, blind.hits, metrics,
                              iterative=res)


def reduce_by_method(sim: Simulation, method=None) -> Reduction:
    """removal.method(simple_pca / iterative_pca)に従って 1 通り解析する"""
    method = sim.cfg.removal.method if method is None else method
    if method == "iterative_pca":
        return reduce_iterative(sim)
    if method == "simple_pca":
        return reduce(sim, sim.cfg.removal.n_components, sim.cfg.removal.mean_mode, label="simple_pca")
    raise ValueError(f"未知の removal.method です: {method}")


def expected_thermal_rms_k(cfg: Config) -> float:
    """熱雑音だけを時間平均したときに期待される RMS [K]"""
    if not cfg.noise.thermal.enabled:
        return 0.0
    sigma = noise.radiometer_sigma_k(
        cfg.noise.thermal.tsys_k, cfg.spectrometer.channel_width_mhz * 1e6, cfg.modulation.dt_s
    )
    return float(sigma / np.sqrt(cfg.n_time))
