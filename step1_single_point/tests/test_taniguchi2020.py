"""Taniguchi et al. (2020) Fig. 6 再現シミュレーター (fmlo_sim/taniguchi2020.py) のテスト"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmlo_sim import taniguchi2020 as tg

CONFIG = Path(__file__).resolve().parent.parent / "config" / "taniguchi2020_fig6.yaml"


@pytest.fixture(scope="module")
def cfg():
    return tg.load_fig6_config(CONFIG)


@pytest.fixture(scope="module")
def run(cfg):
    return tg.run_fig6(cfg)


def test_yaml_matches_code_defaults(cfg):
    """YAML と dataclass 既定値が食い違っていない(片方だけ直す事故の検出)"""
    assert cfg.to_dict() == tg.Fig6Config().to_dict()


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        tg.load_fig6_config(None, {"fm.nope": 1})
    with pytest.raises(ValueError):
        tg.load_fig6_config(None, {"reduction.n_components": "100"})  # K < min(N, D)


def test_zigzag_matches_figure(cfg):
    """図から読み取った形: 周期 50、折り返し点が 2 サンプル続く"""
    xi = tg.zigzag_fm_channels(100, cfg.fm.step_ch, cfg.fm.n_steps, cfg.fm.phase)
    assert xi.min() == 0 and xi.max() == cfg.fm.step_ch * cfg.fm.n_steps
    assert np.array_equal(xi[:50], xi[50:])
    steps = np.abs(np.diff(xi))
    assert set(steps.tolist()) == {0, cfg.fm.step_ch}
    assert np.count_nonzero(steps == 0) == 4          # 100 サンプル中の折り返し 4 回
    # 論文パネル下端 (n=0) から輝線は右 (高 IF ch) へ動く = ξ̄ が減る。11 サンプル目で折り返す
    assert xi[1] < xi[0]
    assert xi[13] == xi[14] == 0


def test_modulation_roundtrip():
    """M(M^-1(X)) = X、M^-1 の未観測要素は NaN(論文 §2.2)"""
    rng = np.random.default_rng(0)
    xi = tg.zigzag_fm_channels(20, 3, 4, 2)
    x = rng.standard_normal((20, 30))
    d = tg.demodulate(x, xi)
    assert d.shape == (20, 30 + 12)
    assert np.isnan(d).sum() == 20 * 12
    assert np.array_equal(tg.modulate(d, xi, 30), x)
    # RF に固定された信号は復調後に同じ列へ揃う
    line = np.zeros(42); line[25] = 1.0
    mod = tg.modulate(np.broadcast_to(line, (20, 42)), xi, 30)
    assert np.array_equal(np.argmax(mod, axis=1), 25 - xi)
    assert np.nanmax(np.abs(np.nan_to_num(tg.demodulate(mod, xi))[:, :25])) == 0


def test_pca_centering_adds_mean_back():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 60)) + 7.0 + np.linspace(0, 3, 60)
    assert np.allclose(tg.pca_correlated(x, 0, True), x.mean(0))
    assert np.allclose(tg.pca_correlated(x, 0, False), 0.0)
    # ランク 1 + 平均の行列は K=1 で厳密に再構成される
    low = np.outer(rng.standard_normal(40), rng.standard_normal(60)) + 5.0
    assert np.allclose(tg.pca_correlated(low, 1, True), low)


def test_spectrum_noise_definition():
    rng = np.random.default_rng(2)
    d = rng.standard_normal((400, 5)) * 2.0 + 1.0
    d[:399, 4] = np.nan                       # hits=1 の列
    spec, noise, hits = tg.make_spectrum(d)
    assert hits.tolist() == [400, 400, 400, 400, 1]
    assert np.allclose(spec[:4], d[:, :4].mean(0))
    assert np.allclose(noise[:4], d[:, :4].std(0) / np.sqrt(400))
    assert np.isinf(noise[4])                 # hits <= 2 は雑音無限大 → モデルに入らない


def test_sigma_cutoff_one_and_two_sided():
    s = np.array([6.0, -6.0, 4.0, 0.0])
    n = np.ones(4)
    assert tg.sigma_cutoff(s, n, 5.0).tolist() == [6.0, 0.0, 0.0, 0.0]
    assert tg.sigma_cutoff(s, n, 5.0, two_sided=True).tolist() == [6.0, -6.0, 0.0, 0.0]


def test_simulation_shapes_and_decomposition(cfg, run):
    truth, res = run
    n, d = cfg.spectrometer.n_time, cfg.spectrometer.n_channels
    assert truth.t_cal.shape == (n, d)
    assert np.allclose(truth.t_cal, truth.t_cor + truth.t_ast + truth.e_nc)
    assert len(res.steps) == 2 and [s.n_components for s in res.steps] == cfg.reduction.n_components
    for s in res.steps:
        assert np.isfinite(s.t_cor).all() and np.isfinite(s.t_ast).all()
        assert np.allclose(truth.t_cal, s.t_cor + s.t_ast + s.e_nc)
    assert np.array_equal(res.steps[0].x_in, truth.t_cal)
    assert np.allclose(res.steps[1].x_in, truth.t_cal - res.steps[0].t_ast)


def test_deterministic(cfg):
    a, _ = tg.run_fig6(cfg)
    b, _ = tg.run_fig6(cfg)
    assert np.array_equal(a.t_cal, b.t_cal)


def test_fig6_behaviour(cfg, run):
    """論文 Fig. 6 が示す定性的挙動: 1 回目は輝線を取りこぼし残差に負の跡、2 回目で改善"""
    truth, res = run
    s1, s2 = res.steps
    amp = cfg.line.amplitude
    assert (s1.model >= 0).all() and (s2.model >= 0).all()      # T^ast に負値なし
    assert 0.55 * amp < s1.model.max() < 0.85 * amp             # 論文図: 3.64 / 5.0
    assert s2.model.max() > 0.9 * amp                           # 論文図: 5.0 / 5.0
    # モデルは輝線の近傍だけ
    nz = np.flatnonzero(s2.model)
    assert np.all(np.abs(nz - cfg.line.center_ch) < 2 * cfg.line.fwhm_ch)
    # 真の相関成分・真の輝線に対する誤差が 2 回目で小さくなる
    err = lambda s: np.sqrt(np.mean((s.t_cor - truth.t_cor) ** 2))
    assert err(s2) < 0.5 * err(s1)
    assert np.abs(s2.model - truth.line_spectrum).max() < np.abs(s1.model - truth.line_spectrum).max()
    # 1 回目の残差には輝線の周りに負の跡が残る(時間平均で有意に負)
    resid1 = np.nanmean(tg.demodulate(s1.e_nc, truth.xi_bar), axis=0)
    resid2 = np.nanmean(tg.demodulate(s2.e_nc, truth.xi_bar), axis=0)
    near = np.abs(np.arange(resid1.size) - cfg.line.center_ch) < 3 * cfg.line.fwhm_ch
    assert resid1[near].min() < -0.5
    assert resid2[near].min() > 0.5 * resid1[near].min()


def test_convergence_mode(cfg):
    truth = tg.simulate_fig6(cfg)
    res = tg.run_iterations(truth.t_cal, truth.xi_bar, [4, 1], convergence_eps=0.05, max_iterations=20)
    assert res.converged is True
    assert 2 < len(res.steps) <= 20
    assert res.steps[-1].variation < 0.05
    res2 = tg.run_iterations(truth.t_cal, truth.xi_bar, [4, 1], convergence_eps=1e-12, max_iterations=3)
    assert res2.converged is False and len(res2.steps) == 3


def test_input_not_modified(cfg):
    truth = tg.simulate_fig6(cfg)
    before = truth.t_cal.copy()
    tg.run_iterations(truth.t_cal, truth.xi_bar, [4, 1])
    assert np.array_equal(truth.t_cal, before)


def test_panel_pixels():
    x = np.arange(2048, dtype=float)[None, :].repeat(3, 0)
    p = tg.to_panel_pixels(x, 720)
    assert p.shape == (3, 720) and p[0, 0] == 1 and p[0, -1] == 2046
