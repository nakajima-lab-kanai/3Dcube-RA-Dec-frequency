import numpy as np
import pytest

from fmlo_sim import units


def test_fwhm_sigma_roundtrip():
    assert units.sigma_to_fwhm(1.0) == pytest.approx(2.354820045)
    assert units.fwhm_to_sigma(units.sigma_to_fwhm(50.0)) == pytest.approx(50.0)


def test_gaussian_half_maximum_at_half_fwhm():
    sigma = 50.0
    half = units.sigma_to_fwhm(sigma) / 2.0
    assert np.exp(-0.5 * (half / sigma) ** 2) == pytest.approx(0.5)


def test_velocity_to_frequency_width():
    # 150 GHz で 50 km/s は 150e3 MHz × 50 / 299792.458 ≈ 25.017 MHz
    dnu_ghz = units.velocity_width_to_freq_width(50.0, 150.0)
    assert dnu_ghz * 1e3 == pytest.approx(25.0173, abs=1e-3)


def test_velocity_frequency_roundtrip():
    dnu = units.velocity_width_to_freq_width(123.0, 140.0)
    assert units.freq_width_to_velocity_width(dnu, 140.0) == pytest.approx(123.0)


def test_freq_to_velocity_sign():
    # radio 定義:周波数が高いほど速度は負(近づく向き)
    assert units.freq_to_velocity(150.0, 150.0) == 0.0
    assert units.freq_to_velocity(150.1, 150.0) < 0.0
