import numpy as np
import pytest
from sklearn.decomposition import PCA

from fmlo_sim import cube as cube_mod
from fmlo_sim import pipeline, removal
from fmlo_sim.config import load_config
from fmlo_sim.units import sigma_to_fwhm

# テストを軽くするための小さな構成
SMALL = {
    "spectrometer.n_channels": 512,
    "cube.pixel_arcsec": 50.0,
    "observation.duration_s": 20.0,
    "modulation.half_width_ch": 60,
}


def test_pca_channel_mean_matches_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 30)) + 5.0 + np.outer(rng.normal(size=80), np.linspace(1, 2, 30))
    pca = PCA(n_components=2).fit(x)
    expected = x - pca.inverse_transform(pca.transform(x))
    result = removal.pca_remove(x, 2, "channel_mean")
    np.testing.assert_allclose(result.cleaned, expected, atol=1e-10)
    np.testing.assert_allclose(result.cleaned + result.removed, x, atol=1e-10)


def test_pca_zero_components_keeps_data():
    x = np.random.default_rng(1).normal(size=(20, 10))
    np.testing.assert_array_equal(removal.pca_remove(x, 0, "none").cleaned, x)


def test_cube_is_gaussian_ball(tmp_path):
    cfg = load_config(overrides=SMALL)
    grid = cube_mod.make_frequency_grid(cfg)
    cube, ra, dec = cube_mod.make_gaussian_cube(cfg, grid)

    assert cube.shape == (ra.size, dec.size, grid.n_wide)
    assert grid.n_wide > grid.n_channels  # キューブは分光計の帯域より広い
    spec, (i_ra, i_dec) = cube_mod.extract_spectrum(cube, ra, dec, 0.0, 0.0)
    assert ra[i_ra] == 0.0 and dec[i_dec] == 0.0
    assert spec.max() == pytest.approx(cfg.source.peak_k, rel=1e-3)
    # 空間方向:σ = 50 arcsec 離れた点で exp(-1/2) 倍
    off, _ = cube_mod.extract_spectrum(cube, ra, dec, 50.0, 0.0)
    assert off.max() / spec.max() == pytest.approx(np.exp(-0.5), rel=1e-4)
    # 周波数方向:半値全幅が 50 km/s 相当
    above = grid.freq_ghz[spec >= 0.5 * spec.max()]
    fwhm_ghz = above[-1] - above[0] + grid.channel_width_ghz
    expected = 150.0 * 50.0 / 299792.458
    assert fwhm_ghz == pytest.approx(expected, abs=2 * grid.channel_width_ghz)
    assert sigma_to_fwhm(cfg.source.sigma_arcsec) == pytest.approx(117.74, abs=0.01)

    path = tmp_path / "cube.npz"
    cube_mod.save_cube(path, cube, ra, dec, grid.freq_ghz, cfg)
    cube2, ra2, _, freq2, meta = cube_mod.load_cube(path)
    np.testing.assert_array_equal(cube2, cube)
    np.testing.assert_array_equal(freq2, grid.freq_ghz)
    assert meta["units"]["cube"] == "K"


def test_noiseless_pipeline_restores_truth():
    """雑音なし・PCA なしなら復元スペクトルは正解と一致する"""
    cfg = load_config(overrides={
        **SMALL, "noise.thermal.enabled": False, "noise.atmosphere.enabled": False,
    })
    sim = pipeline.simulate(cfg)
    red = pipeline.reduce(sim, 0, "none")
    assert red.metrics["peak_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert red.metrics["integrated_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert red.metrics["line_free_residual_rms_k"] < 1e-9


def test_same_seed_reproduces_and_modes_share_data():
    cfg = load_config(overrides=SMALL)
    a, b = pipeline.simulate(cfg), pipeline.simulate(cfg)
    np.testing.assert_array_equal(a.observed, b.observed)
    results = pipeline.reduce_all_modes(a)
    assert set(results) == {"none", "channel_mean", "no_pca"}
    # 熱雑音は PCA で消えないので、残差 RMS は熱雑音の期待値より小さくはならない
    assert results["none"].metrics["line_free_residual_rms_k"] > 0.8 * pipeline.expected_thermal_rms_k(cfg)


@pytest.mark.parametrize("mode", ["none", "channel_mean"])
def test_nonzero_atmosphere_offset_runs(mode):
    """大気の定数オフセットが 0 以外でも最後まで動く"""
    cfg = load_config(overrides={**SMALL, "noise.atmosphere.offset_k": 50.0})
    red = pipeline.reduce(pipeline.simulate(cfg), 2, mode)
    assert np.isfinite(red.metrics["line_free_residual_rms_k"])
    assert np.isfinite(red.metrics["peak_ratio"])
