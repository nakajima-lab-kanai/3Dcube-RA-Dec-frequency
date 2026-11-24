"""従来 OTF(LO 固定・OFF 点を引く)の模擬観測・処理のテスト"""
from __future__ import annotations

import numpy as np
import pytest

from fmlo_sim import evaluation, mapping, otf, pipeline, scan
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config


def small_cfg(**extra):
    sets = {"scan.mode": "x_scan", "scan.map_size_arcsec": 40.0, "scan.scan_speed_arcsec_s": 20.0,
            "scan.row_spacing_arcsec": 10.0, "cube.half_extent_arcsec": 60.0, "cube.pixel_arcsec": 4.0,
            "source.sigma_arcsec": 15.0, "spectrometer.n_channels": 256, "modulation.half_width_ch": 30,
            "otf.off_integration_s": 1.0, **extra}
    return load_config(DEFAULT_CONFIG_PATH, sets)


def test_default_config_has_otf_defaults():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.otf.off_interval_rows == 1 and cfg.otf.baseline_order == 1


def test_timeline_brackets_every_group_with_off():
    p = scan.raster_pass("x", 40.0, 20.0, 10.0, 0.1)
    tl = otf.make_timeline(p, off_interval_rows=2, n_off=3, dt_s=0.1)
    n_rows = int(p.row.max()) + 1
    assert tl.n_off_blocks == int(np.ceil(n_rows / 2)) + 1
    assert tl.n_total == p.n_time + 3 * tl.n_off_blocks
    on = tl.on_index[tl.on_index >= 0]
    np.testing.assert_array_equal(on, np.arange(p.n_time))      # ON は軌跡の順のまま
    assert tl.off_block[0] == 0 and tl.off_block[-1] == tl.n_off_blocks - 1


def test_off_interpolation_weights():
    p = scan.raster_pass("x", 40.0, 20.0, 10.0, 0.1)
    tl = otf.make_timeline(p, 1, 2, 0.1)
    b0, b1, w = otf.off_weights(tl, "linear")
    assert np.all(b1 == b0 + 1) and np.all((w > 0) & (w < 1))
    _, _, wn = otf.off_weights(tl, "nearest")
    assert set(np.unique(wn)) <= {0.0, 1.0}


def test_polynomial_baseline_removes_line_free_polynomial():
    x = np.linspace(-1, 1, 64)
    line = np.exp(-0.5 * (x / 0.05) ** 2)   # 輝線のないチャンネル(|x| > 0.4)では 0 とみなせる
    spec = np.stack([3 + 2 * x + line, -1 + 0.5 * x + line])
    free = np.abs(x) > 0.4
    cal, _ = otf.polynomial_baseline(spec, free, 1)
    np.testing.assert_allclose(cal, np.stack([line, line]), atol=1e-6)
    same, base = otf.polynomial_baseline(spec, free, -1)
    np.testing.assert_array_equal(same, spec)
    assert not base.any()


def test_noise_free_otf_equals_reference():
    """雑音なしなら ON − OFF = 天体信号(OFF は 0)。グリッディング結果は基準と一致"""
    cfg = small_cfg(**{"noise.thermal.enabled": False, "noise.atmosphere.enabled": False,
                       "otf.baseline_order": -1})
    sky = pipeline.make_internal_sky_cube(cfg)
    p = scan.make_passes(cfg)[0]
    o = otf.observe_pass(sky, cfg, p, 0)
    r = otf.reduce_pass(o, np.ones(o.grid.n_channels, dtype=bool), "linear", -1)
    np.testing.assert_allclose(r.calibrated, o.signal_on, atol=1e-12)


def test_offset_atmosphere_is_removed_by_off_and_baseline():
    """一定 + 傾きの大気は ON − OFF と 1 次ベースラインで消え、熱雑音だけが残る"""
    cfg = small_cfg(**{"noise.atmosphere.rms_k": 0.0, "noise.atmosphere.offset_k": 50.0})
    sky = pipeline.make_internal_sky_cube(cfg)
    p = scan.make_passes(cfg)[0]
    o = otf.observe_pass(sky, cfg, p, 0)
    grid = o.grid
    _, free = evaluation.line_masks(grid.freq_ghz, cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms, 2.0)
    free_ch = evaluation.line_free_channels(free, o.shifts_ch, grid.start0, grid.n_channels)
    r = otf.reduce_pass(o, free_ch, "linear", 1)
    resid = r.calibrated - o.signal_on
    sigma = o.noise.components["thermal"].std()
    assert abs(resid.mean()) < 0.05 * sigma
    assert resid.std() < 1.1 * sigma


def test_otf_reproducible_and_independent_of_fmlo_streams():
    cfg = small_cfg()
    sky = pipeline.make_internal_sky_cube(cfg)
    p = scan.make_passes(cfg)[0]
    a, b = otf.observe_pass(sky, cfg, p, 0), otf.observe_pass(sky, cfg, p, 0)
    np.testing.assert_array_equal(a.raw, b.raw)
    f = scan.observe_pass(sky, cfg, p, 0)
    n = min(5, p.n_time)
    assert not np.allclose(a.noise.components["thermal"][:n], f.noise.components["thermal"][:n])


def test_invalid_otf_settings_raise():
    with pytest.raises(ValueError):
        small_cfg(**{"otf.off_interpolation": "cubic"})
    with pytest.raises(ValueError):
        small_cfg(**{"otf.off_interval_rows": 0})


@pytest.mark.parametrize("name", ["default.yaml", "step2_scan.yaml", "step2_sweep.yaml"])
def test_shipped_configs_load(name):
    """同梱の設定ファイルがすべて読み込める(未知のキーがない)"""
    from pathlib import Path

    from fmlo_sim.config import config_from_dict, read_yaml

    data = read_yaml(Path(DEFAULT_CONFIG_PATH).parent / name)
    data.pop("sweep", None)   # run_step2_sweep.py が取り除いてから読む
    config_from_dict(data)
