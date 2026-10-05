import numpy as np
import pytest

from fmlo_sim import noise


def test_thermal_noise_matches_radiometer_equation():
    tsys, dnu_hz, dt = 200.0, 488281.25, 0.1
    expected = tsys / np.sqrt(dnu_hz * dt)
    assert noise.radiometer_sigma_k(tsys, dnu_hz, dt) == pytest.approx(expected)

    tod = noise.thermal_noise(500, 400, tsys, dnu_hz, dt, np.random.default_rng(0))
    # 20万サンプルなので標準偏差の統計誤差は約 0.16 %
    assert tod.std() == pytest.approx(expected, rel=0.01)
    assert abs(tod.mean()) < 0.01 * expected
    # チャンネル間は独立
    assert abs(np.corrcoef(tod[:, 0], tod[:, 1])[0, 1]) < 0.15


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_colored_noise_slope(alpha):
    """有色雑音のパワースペクトルの傾きが指定した alpha とおおむね一致する"""
    n, dt = 4096, 0.1
    rng = np.random.default_rng(0)
    freqs = np.fft.rfftfreq(n, dt)[1:]
    power = np.zeros(freqs.size)
    n_real = 50
    for _ in range(n_real):
        x = noise.colored_noise_timeseries(n, dt, alpha, 1.0, rng)
        power += np.abs(np.fft.rfft(x))[1:] ** 2
    slope = np.polyfit(np.log10(freqs), np.log10(power / n_real), 1)[0]
    assert slope == pytest.approx(-alpha, abs=0.1)


def test_colored_noise_rms_and_mean():
    x = noise.colored_noise_timeseries(600, 0.1, 1.0, 2.5, np.random.default_rng(3))
    assert x.std() == pytest.approx(2.5)
    assert abs(x.mean()) < 1e-12


def test_atmosphere_is_fixed_in_sky_frequency():
    """大気の周波数方向の形は空の周波数に固定され、窓と一緒に切り出される"""
    n_ch, start0 = 32, 5
    n_wide = n_ch + 2 * start0
    freq = 150.0 + (np.arange(n_wide) - n_wide / 2) * 1e-3
    shape = noise.atmosphere_shape(freq, 150.0, n_ch * 1e-3, 0.1)
    shifts = np.array([0, 3, -3])
    series = np.array([1.0, 2.0, -1.0])
    tod = noise.atmosphere_noise(series, shape, shifts, start0, n_ch, offset_k=10.0)
    for t in range(3):
        window = shape[start0 + shifts[t]:start0 + shifts[t] + n_ch]
        np.testing.assert_allclose(tod[t], (10.0 + series[t]) * window)


def test_noise_components_container():
    comps = noise.NoiseComponents((4, 8))
    comps.add("a", np.ones((4, 8)))
    comps.add("b", 2 * np.ones((4, 8)))
    np.testing.assert_array_equal(comps.total(), 3 * np.ones((4, 8)))
    comps.add("a", np.zeros((4, 8)))  # 同じ名前で差し替え
    np.testing.assert_array_equal(comps.total(), 2 * np.ones((4, 8)))
    with pytest.raises(ValueError):
        comps.add("bad", np.ones((4, 7)))
