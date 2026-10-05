import numpy as np
import pytest

from fmlo_sim import modulation

N_CH = 64
HALF = 10
START0 = HALF + 3
N_WIDE = N_CH + 2 * START0


def _wide_spectrum():
    x = np.arange(N_WIDE)
    return np.exp(-0.5 * ((x - N_WIDE / 2) / 4.0) ** 2) + 0.01 * x


@pytest.mark.parametrize("pattern", ["zigzag", "random"])
def test_roundtrip_without_noise(pattern):
    """雑音なしなら、変調→復調で元のスペクトルに戻る"""
    rng = np.random.default_rng(0)
    shifts = modulation.make_shift_pattern(pattern, 200, HALF, 3, rng)
    wide = _wide_spectrum()
    tod = modulation.modulate(wide, shifts, START0, N_CH)
    restored, hits = modulation.demodulate(tod, shifts, START0, N_WIDE)

    assert tod.shape == (200, N_CH)
    covered = hits > 0
    np.testing.assert_allclose(restored[covered], wide[covered], rtol=0, atol=1e-12)
    assert np.all(np.isnan(restored[~covered]))
    # 余白(3 ch)の部分は一度も観測されない
    assert not covered[:3].any() and not covered[-3:].any()


def test_no_wraparound():
    """端の信号が反対側に回り込まない(np.roll との違い)"""
    wide = np.zeros(N_WIDE)
    wide[START0 + N_CH - 1] = 1.0  # シフト 0 の窓の上端
    tod = modulation.modulate(wide, np.array([0, 1, -1]), START0, N_CH)
    assert tod[0, -1] == 1.0
    assert tod[1, -2] == 1.0
    assert tod[2].sum() == 0.0  # 窓の外に出た信号は消える。先頭には現れない


def test_zigzag_pattern():
    shifts = modulation.make_shift_pattern("zigzag", 100, 6, 2)
    assert shifts.dtype.kind == "i"
    np.testing.assert_array_equal(shifts[:13], [0, 2, 4, 6, 4, 2, 0, -2, -4, -6, -4, -2, 0])
    assert shifts.min() == -6 and shifts.max() == 6


def test_random_pattern_range_and_seed():
    a = modulation.make_shift_pattern("random", 1000, 5, rng=np.random.default_rng(1))
    b = modulation.make_shift_pattern("random", 1000, 5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert a.min() == -5 and a.max() == 5


def test_window_out_of_range_raises():
    with pytest.raises(ValueError):
        modulation.modulate(_wide_spectrum(), np.array([START0 + 1]), START0, N_CH)
    with pytest.raises(ValueError):
        modulation.modulate(_wide_spectrum(), np.array([-START0 - 1]), START0, N_CH)


def test_time_varying_spectrum():
    """時刻ごとに違うスペクトル(段階2以降のスキャン用)も切り出せる"""
    wide = np.arange(3 * N_WIDE, dtype=float).reshape(3, N_WIDE)
    shifts = np.array([0, 2, -2])
    tod = modulation.modulate(wide, shifts, START0, N_CH)
    for t in range(3):
        np.testing.assert_array_equal(tod[t], wide[t, START0 + shifts[t]:START0 + shifts[t] + N_CH])
