"""段階 2(ノイズモデル拡張・反復 PCA・1 ビーム OTF スキャン)のテスト"""
from __future__ import annotations

import numpy as np
import pytest

from fmlo_sim import cube as cube_mod
from fmlo_sim import evaluation, mapping, modulation, noise, pipeline, removal, scan
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config, spawn_rngs
from fmlo_sim.units import fwhm_to_sigma


# ---------------------------------------------------------------------------
# 後方互換
# ---------------------------------------------------------------------------

def test_default_config_new_keys_keep_old_behaviour():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.removal.method == "simple_pca"
    assert cfg.modulation.cycle_jitter is False and cfg.modulation.max_abs_shift_ch == cfg.modulation.half_width_ch
    assert cfg.noise.instrument.enabled is False
    assert cfg.telescope.beam_convolve is False


def test_spawn_rngs_existing_streams_unchanged():
    """成分を末尾に足しても、既存 3 成分の乱数列は変わらない"""
    old = np.random.SeedSequence(123).spawn(3)
    new = spawn_rngs(123)
    for name, ss in zip(("modulation", "thermal", "atmosphere"), old):
        assert np.array_equal(new[name].normal(size=5), np.random.default_rng(ss).normal(size=5))


def test_pass_rngs_are_independent_of_point_streams():
    a, b, c = spawn_rngs(1), spawn_rngs(1, 0), spawn_rngs(1, 1)
    xa, xb, xc = (r["thermal"].normal(size=8) for r in (a, b, c))
    assert not np.allclose(xa, xb) and not np.allclose(xb, xc)
    assert np.array_equal(spawn_rngs(1, 1)["thermal"].normal(size=8), xc)


# ---------------------------------------------------------------------------
# マイルストーン 1:雑音モデル・PSD
# ---------------------------------------------------------------------------

def test_random_walk_mean_and_rms():
    s = noise.random_walk_timeseries(1000, 0.7, np.random.default_rng(0))
    assert abs(s.mean()) < 1e-12 and np.isclose(s.std(), 0.7)


def _small_grid():
    cfg = load_config(DEFAULT_CONFIG_PATH, {"spectrometer.n_channels": 128, "modulation.half_width_ch": 10,
                                            "modulation.zigzag_step_ch": 2, "cube.margin_ch": 5,
                                            "observation.duration_s": 5.0})
    return cfg, cube_mod.make_frequency_grid(cfg)


def test_external_atmosphere_1d_and_2d(tmp_path):
    cfg, grid = _small_grid()
    n = cfg.n_time
    series = np.linspace(-1, 1, n)
    np.save(tmp_path / "a.npy", series)
    assert np.array_equal(noise.load_external_atmosphere(tmp_path / "a.npy", n, grid.n_wide), series)

    wide = np.random.default_rng(0).normal(size=(n, grid.n_wide))
    np.savez(tmp_path / "b.npz", atmosphere=wide, freq_ghz=grid.freq_ghz)
    over = {"noise.atmosphere.model": "external", "noise.atmosphere.external_path": str(tmp_path / "b.npz"),
            "noise.thermal.enabled": False}
    cfg2 = load_config(DEFAULT_CONFIG_PATH, {"spectrometer.n_channels": 128, "modulation.half_width_ch": 10,
                                             "modulation.zigzag_step_ch": 2, "cube.margin_ch": 5,
                                             "observation.duration_s": 5.0, **over})
    shifts = modulation.shifts_from_config(cfg2.modulation, n, None)
    comps = noise.make_noise_components(cfg2.noise, cfg2.spectrometer, grid, shifts, 0.1, spawn_rngs(0))
    assert np.array_equal(comps.components["atmosphere"], modulation.modulate(wide, shifts, grid.start0, 128))

    np.save(tmp_path / "bad.npy", np.zeros((n, 7)))
    with pytest.raises(ValueError):
        noise.load_external_atmosphere(tmp_path / "bad.npy", n, grid.n_wide)
    np.savez(tmp_path / "badf.npz", atmosphere=wide, freq_ghz=grid.freq_ghz + 1.0)
    with pytest.raises(ValueError):
        noise.load_external_atmosphere(tmp_path / "badf.npz", n, grid.n_wide, grid.freq_ghz)


def test_external_requires_path():
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG_PATH, {"noise.atmosphere.model": "external"})


def test_instrument_noise_is_channel_fixed():
    tod = noise.instrument_noise(500, 16, 0.1, 1.0, 0.2, np.random.default_rng(1))
    assert np.allclose(tod, tod[:, :1]) and np.isclose(tod[:, 0].std(), 0.2)


def test_instrument_noise_does_not_change_other_components():
    base = {"noise.instrument.enabled": False}
    a = pipeline.simulate(load_config(DEFAULT_CONFIG_PATH, base))
    b = pipeline.simulate(load_config(DEFAULT_CONFIG_PATH, {"noise.instrument.enabled": True}))
    for name in ("thermal", "atmosphere"):
        assert np.array_equal(a.noise.components[name], b.noise.components[name])
    assert "instrument" in b.noise.components and "instrument" not in a.noise.components


def test_psd_white_noise_matches_expected_level():
    rng = np.random.default_rng(3)
    sigma, dt = 0.9, 0.1
    x = rng.normal(0, sigma, size=(4000, 64))
    cmp = evaluation.psd_comparison({"white": x}, dt, np.ones(64, bool), sigma, low_freq_hz=0.5)
    assert abs(cmp["excess"]["white"]["all_freq_per_channel"] - 1.0) < 0.03
    assert abs(cmp["excess"]["white"]["all_freq_common"] - 1.0) < 0.1
    assert np.isclose(cmp["white_per_channel"], 2 * sigma ** 2 * dt)


# ---------------------------------------------------------------------------
# マイルストーン 2:変調・反復 PCA
# ---------------------------------------------------------------------------

def test_cycle_jitter_properties():
    a, step, n = 20, 4, 1000
    rng = np.random.default_rng(5)
    base = modulation.make_shift_pattern("zigzag", n, a, step)
    jit = modulation.make_shift_pattern("zigzag", n, a, step, rng, cycle_jitter=True)
    off = jit - base
    period, _ = modulation.compute_zigzag_period_samples(a, step, 1.0)
    assert np.abs(off).max() <= step // 2 and np.abs(jit).max() <= a + step // 2
    for c in range(n // period):
        assert np.unique(off[c * period:(c + 1) * period]).size == 1
    assert np.unique(off[::period]).size > 1
    with pytest.raises(ValueError):
        modulation.make_shift_pattern("random", n, a, step, rng, cycle_jitter=True)


def test_cycle_jitter_widens_grid():
    cfg = load_config(DEFAULT_CONFIG_PATH, {"modulation.cycle_jitter": True})
    grid = cube_mod.make_frequency_grid(cfg)
    assert grid.start0 == cfg.modulation.half_width_ch + cfg.modulation.zigzag_step_ch // 2 + cfg.cube.margin_ch
    pipeline.simulate(cfg)   # 窓がグリッドからはみ出さない


def test_demodulate_tod_weights_match_demodulate():
    rng = np.random.default_rng(0)
    n_time, n_ch, start0, n_wide = 50, 32, 8, 48
    shifts = rng.integers(-8, 9, size=n_time)
    tod = rng.normal(size=(n_time, n_ch))
    demod, w = modulation.demodulate_tod(tod, shifts, start0, n_wide)
    spec, hits = modulation.demodulate(tod, shifts, start0, n_wide)
    assert np.array_equal(w.sum(axis=0), hits)
    assert np.all(np.isnan(demod[w == 0])) and np.all(np.isfinite(demod[w == 1]))
    mean = np.where(hits > 0, np.nansum(demod, axis=0) / np.maximum(hits, 1), np.nan)
    assert np.allclose(mean, spec, equal_nan=True)


def test_iterative_pca_recovers_line_better_than_simple():
    over = {"noise.atmosphere.offset_k": 50.0, "removal.n_components": 2, "removal.method": "iterative_pca"}
    cfg = load_config(DEFAULT_CONFIG_PATH, over)
    sim = pipeline.simulate(cfg)
    simple = pipeline.reduce(sim, 2, "none")
    it = pipeline.reduce_iterative(sim)
    assert it.metrics["integrated_ratio"] > 0.9 > simple.metrics["integrated_ratio"]
    res = it.iterative
    assert res.converged and res.n_iter >= 2 and res.rel_change[-1] < cfg.removal.tol
    # 恒等式 X = C + (X - C)
    assert np.allclose(res.removed[0] + res.cleaned[0], sim.observed)


def test_iterative_pca_stops_at_max_iter():
    xs = [np.random.default_rng(0).normal(size=(40, 16))]
    calls = []

    def model_fn(cleaned):
        calls.append(1)
        return [c * (0.5 + 0.1 * len(calls)) for c in cleaned]   # 収束しないモデル

    res = removal.iterative_pca(xs, 1, "none", model_fn, max_iter=4, tol=1e-12)
    assert res.n_iter == 4 and not res.converged and res.rel_change[0] is None


def test_reduce_by_method_simple_matches_reduce():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    sim = pipeline.simulate(cfg)
    a = pipeline.reduce_by_method(sim, "simple_pca")
    b = pipeline.reduce(sim, cfg.removal.n_components, cfg.removal.mean_mode)
    assert np.array_equal(a.restored_k, b.restored_k, equal_nan=True)


# ---------------------------------------------------------------------------
# マイルストーン 3:ビーム・補間・スキャン・マップ
# ---------------------------------------------------------------------------

def test_beam_convolution_matches_gaussian_theory():
    pix, sig_src, fwhm_b = 2.0, 20.0, 12.0
    axis = np.arange(-60, 61) * pix
    img = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2) / sig_src ** 2)[:, :, None]
    out = cube_mod.convolve_beam(img, pix, fwhm_b)
    sb = fwhm_to_sigma(fwhm_b)
    expected_peak = sig_src ** 2 / (sig_src ** 2 + sb ** 2)
    assert abs(out.max() - expected_peak) < 2e-3
    assert np.isclose(out.sum(), img.sum(), rtol=1e-6)
    with pytest.raises(ValueError):
        cube_mod.convolve_beam(img, 10.0, 12.0)


def test_bilinear_exact_for_linear_field():
    ra, dec = np.arange(-5, 6) * 3.0, np.arange(-4, 5) * 2.0
    field = (2 * ra[:, None] - 0.5 * dec[None, :] + 1.0)[:, :, None]
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-15, 15, 50), rng.uniform(-8, 8, 50)
    got = cube_mod.sample_spectra(field, ra, dec, x, y)[:, 0]
    assert np.allclose(got, 2 * x - 0.5 * y + 1.0)
    with pytest.raises(ValueError):
        cube_mod.sample_spectra(field, ra, dec, [16.0], [0.0])


def test_raster_pass_geometry():
    p = scan.raster_pass("x", 100.0, 20.0, 10.0, 0.1)
    n_along, n_rows = 51, 11
    assert p.n_time == n_along * n_rows
    assert p.ra_arcsec.min() == -50 and p.ra_arcsec.max() == 50
    row0, row1 = p.ra_arcsec[:n_along], p.ra_arcsec[n_along:2 * n_along]
    assert np.allclose(np.diff(row0), 2.0) and np.allclose(row1, row0[::-1])
    assert np.allclose(np.unique(p.dec_arcsec), np.arange(-50, 51, 10))
    q = scan.raster_pass("y", 100.0, 20.0, 10.0, 0.1)
    assert np.array_equal(q.ra_arcsec, p.dec_arcsec) and np.array_equal(q.dec_arcsec, p.ra_arcsec)


def _scan_cfg(**extra):
    over = {
        "source.sigma_arcsec": 15.0, "source.peak_k": 3.0,
        "cube.half_extent_arcsec": 60.0, "cube.pixel_arcsec": 4.0, "cube.margin_ch": 5,
        "spectrometer.n_channels": 256, "modulation.half_width_ch": 30, "modulation.zigzag_step_ch": 6,
        "modulation.cycle_jitter": True,
        "telescope.beam_convolve": True, "telescope.beam_fwhm_arcsec": 12.0,
        "scan.mode": "basket_weave", "scan.map_size_arcsec": 60.0, "scan.scan_speed_arcsec_s": 40.0,
        "scan.row_spacing_arcsec": 6.0, "mapping.grid_pixel_arcsec": 6.0, "mapping.kernel_fwhm_arcsec": 9.0,
        "removal.n_components": 2, "noise.atmosphere.offset_k": 50.0,
    }
    over.update(extra)
    return load_config(DEFAULT_CONFIG_PATH, over)


def _observe(cfg):
    sky = pipeline.make_internal_sky_cube(cfg)
    passes = scan.make_passes(cfg)
    mg = mapping.make_map_grid(cfg)
    obs = [scan.observe_pass(sky, cfg, p, i) for i, p in enumerate(passes)]
    geoms = [mapping.PassGeometry(p.name, p.ra_arcsec, p.dec_arcsec, o.shifts_ch,
                                  mapping.gridding_matrix(mg, p.ra_arcsec, p.dec_arcsec, 9.0, 1.5))
             for p, o in zip(passes, obs)]
    return sky, passes, mg, obs, geoms


def test_scan_observation_reproducible_and_passes_independent():
    cfg = _scan_cfg()
    _, _, _, a, _ = _observe(cfg)
    _, _, _, b, _ = _observe(cfg)
    assert len(a) == 2
    for x, y in zip(a, b):
        assert np.array_equal(x.observed, y.observed)
    na = a[0].noise.components["thermal"]
    nb = a[1].noise.components["thermal"][:na.shape[0]]
    assert not np.allclose(na, nb)


def test_gridding_noise_free_no_removal_equals_reference():
    """雑音なし・除去なしなら、観測データのグリッディング = 天体信号のグリッディング"""
    cfg = _scan_cfg(**{"noise.thermal.enabled": False, "noise.atmosphere.enabled": False,
                       "removal.n_components": 0, "removal.method": "simple_pca"})
    sky, passes, mg, obs, geoms = _observe(cfg)
    g = obs[0].grid
    red = mapping.reduce_passes("x", [obs[0].observed], [geoms[0]], mg, g.start0, g.n_wide, cfg.removal)
    ref = mapping.grid_tod(obs[0].signal_tod, obs[0].shifts_ch, g.start0, g.n_wide, geoms[0].wmat, mg.shape)
    assert np.allclose(red.gridded.data, ref.data, equal_nan=True)
    # 中心画素のスペクトルは、正解(ビーム畳み込み)をカーネルで平滑化した分だけ低いが、近い
    ic = mg.shape[0] // 2
    truth = cube_mod.sample_spectra(sky.data, sky.ra_arcsec, sky.dec_arcsec, [0.0], [0.0])[0]
    k = int(np.argmax(truth))
    assert 0.8 < ref.data[ic, ic, k] / truth[k] <= 1.0


def test_gridding_and_projection_of_constant():
    cfg = _scan_cfg()
    _, passes, mg, obs, geoms = _observe(cfg)
    g = obs[0].grid
    const = np.full_like(obs[0].observed, 2.5)
    cube = mapping.grid_tod(const, obs[0].shifts_ch, g.start0, g.n_wide, geoms[0].wmat, mg.shape).data
    assert np.allclose(cube[np.isfinite(cube)], 2.5)
    proj = mapping.project_cube(np.full((*mg.shape, g.n_wide), 1.5), mg, passes[0].ra_arcsec,
                                passes[0].dec_arcsec, obs[0].shifts_ch, g.start0, g.n_channels)
    assert np.allclose(proj, 1.5)


def test_fourier_basket_weave():
    rng = np.random.default_rng(0)
    sky = rng.normal(size=(32, 32, 2))
    assert np.allclose(mapping.combine_fourier(sky, sky), sky)
    stripes_x = sky + rng.normal(size=(1, 32, 1)) * 3.0     # x_scan の行ごとのオフセット(Dec だけの関数)
    stripes_y = sky + rng.normal(size=(32, 1, 1)) * 3.0     # y_scan の列ごとのオフセット
    fw = mapping.combine_fourier(stripes_x, stripes_y)
    mean = (stripes_x + stripes_y) / 2
    assert np.std(fw - sky) < 0.5 * np.std(mean - sky)


def test_scan_iterative_improves_flux_over_simple():
    cfg_s = _scan_cfg(**{"removal.method": "simple_pca", "scan.mode": "x_scan"})
    cfg_i = _scan_cfg(**{"removal.method": "iterative_pca", "scan.mode": "x_scan", "removal.cutoff_sigma": 3.0})
    sky, passes, mg, obs, geoms = _observe(cfg_s)
    g = obs[0].grid
    ref = mapping.grid_tod(obs[0].signal_tod, obs[0].shifts_ch, g.start0, g.n_wide, geoms[0].wmat, mg.shape).data
    in_line, _ = evaluation.line_masks(g.freq_ghz, cfg_s.source.center_freq_ghz, cfg_s.source.line_fwhm_kms, 2.0)
    out = {}
    for name, cfg in (("simple", cfg_s), ("iter", cfg_i)):
        red = mapping.reduce_passes("x", [obs[0].observed], [geoms[0]], mg, g.start0, g.n_wide, cfg.removal)
        m0 = evaluation.moment0_map(red.gridded.data, in_line, g.channel_width_ghz, 150.0)
        r0 = evaluation.moment0_map(ref, in_line, g.channel_width_ghz, 150.0)
        out[name] = np.nansum(m0) / np.nansum(r0)
    assert out["iter"] > out["simple"]
