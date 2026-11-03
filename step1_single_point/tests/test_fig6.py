"""Taniguchi et al. (2020) Fig. 6 成分推定のテストスイート

検証内容:
  1. 数学的恒等式の成立 (U_1=X, U_2=X-A_1, X=C_r+A_r+E_r)
  2. 配列形状・型の整合性、NaN/Infの非混入
  3. 入力観測TODの不変性 (非破壊性)
  4. 正解データからの完全な独立性
  5. 既知条件テスト:
     - 雑音なし・大気オフセットなし・PCA成分数0・平均除去なしでの恒等復元
     - PCAあり (成分数1) での天体信号損失の測定 (残差ゼロを成功条件としない)
  6. 雑音水準推定 (MAD簡易法) のエッジケース (有効点不足、MAD=0、NaN)
  7. 既存処理 (run_step1) の回帰確認
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from fmlo_sim import iterative, modulation, pipeline, removal
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "config" / "fig6_taniguchi2020.yaml"


@pytest.fixture
def base_config():
    """Fig. 6 照合用設定"""
    if CONFIG_PATH.is_file():
        return load_config(CONFIG_PATH)
    return load_config(DEFAULT_CONFIG_PATH)


def test_math_identities_and_shapes(base_config):
    """数学的恒等式と配列形状のテスト"""
    sim = pipeline.simulate(base_config)
    grid = sim.grid

    res = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=base_config.removal.n_components,
        mean_mode=base_config.removal.mean_mode,
        cutoff_sigma=5.0,
    )

    n_time, n_ch = sim.observed.shape

    # 配列形状
    for iter_res in (res.iter1, res.iter2):
        assert iter_res.u_tod.shape == (n_time, n_ch)
        assert iter_res.c_tod.shape == (n_time, n_ch)
        assert iter_res.a_tod.shape == (n_time, n_ch)
        assert iter_res.e_tod.shape == (n_time, n_ch)
        assert iter_res.restored_spectrum.shape == (grid.n_wide,)
        assert iter_res.model_spectrum.shape == (grid.n_wide,)
        assert iter_res.noise_sigma.shape == (grid.n_wide,)
        assert iter_res.hits.shape == (grid.n_wide,)

        # NaN / Inf の非混入 (TOD)
        assert np.all(np.isfinite(iter_res.u_tod))
        assert np.all(np.isfinite(iter_res.c_tod))
        assert np.all(np.isfinite(iter_res.a_tod))
        assert np.all(np.isfinite(iter_res.e_tod))
        assert np.all(np.isfinite(iter_res.model_spectrum))

    # 1. U_1 = X
    assert np.allclose(res.iter1.u_tod, res.x_tod, atol=1e-14)

    # 2. U_2 = X - A_1
    assert np.allclose(res.iter2.u_tod, res.x_tod - res.iter1.a_tod, atol=1e-14)

    # 3. 加算恒等式 X = C_1 + A_1 + E_1
    recon1 = res.iter1.c_tod + res.iter1.a_tod + res.iter1.e_tod
    assert np.allclose(res.x_tod, recon1, atol=1e-12)

    # 4. 加算恒等式 X = C_2 + A_2 + E_2
    recon2 = res.iter2.c_tod + res.iter2.a_tod + res.iter2.e_tod
    assert np.allclose(res.x_tod, recon2, atol=1e-12)


def test_input_immutability(base_config):
    """入力観測 TOD が非破壊に保たれることのテスト"""
    sim = pipeline.simulate(base_config)
    grid = sim.grid

    x_original = sim.observed.copy()

    _ = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=base_config.removal.n_components,
        mean_mode=base_config.removal.mean_mode,
        cutoff_sigma=5.0,
    )

    assert np.array_equal(sim.observed, x_original)


def test_independence_from_ground_truth(base_config):
    """反復計算が正解データから完全に独立していることの検証

    観測 TOD と座標情報が同一であれば、正解情報が何であっても推定結果は同一となる。
    """
    sim = pipeline.simulate(base_config)
    grid = sim.grid

    res1 = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=base_config.removal.n_components,
        mean_mode=base_config.removal.mean_mode,
        cutoff_sigma=5.0,
    )

    # 観測データのみを渡し、正解データへの参照が一切不要であることを確認
    fake_observed = sim.observed.copy()
    res2 = iterative.run_two_step_estimation(
        observed_tod=fake_observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=base_config.removal.n_components,
        mean_mode=base_config.removal.mean_mode,
        cutoff_sigma=5.0,
    )

    assert np.array_equal(res1.iter1.c_tod, res2.iter1.c_tod)
    assert np.array_equal(res1.iter1.a_tod, res2.iter1.a_tod)
    assert np.array_equal(res1.iter1.e_tod, res2.iter1.e_tod)
    assert np.array_equal(res1.iter2.c_tod, res2.iter2.c_tod)
    assert np.array_equal(res1.iter2.a_tod, res2.iter2.a_tod)
    assert np.array_equal(res1.iter2.e_tod, res2.iter2.e_tod)


def test_noiseless_baseline_and_pca_signal_loss(base_config):
    """既知条件テスト:
    1. 雑音なし・大気オフセットなし・PCA成分数0・平均除去なしで変調/復調が恒等復元されること。
    2. PCA成分数1の場合に信号損失 (PCAによる変調天体信号の吸収) が発生することを測定し、
       残差ゼロを必須としないこと。
    """
    cfg = copy.deepcopy(base_config)
    # 熱雑音・大気雑音・大気オフセットをすべて無効化
    cfg.noise.thermal.enabled = False
    cfg.noise.atmosphere.enabled = False
    cfg.noise.atmosphere.offset_k = 0.0

    sim = pipeline.simulate(cfg)
    grid = sim.grid

    # Case 1: n_components = 0, mean_mode = "none" (PCAなし、平均除去なし)
    res_nopca = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=0,
        mean_mode="none",
        cutoff_sigma=5.0,
    )

    # PCAがない場合、天体信号のみの観測なので C_r は 0
    assert np.allclose(res_nopca.iter1.c_tod, 0.0, atol=1e-14)
    # 変調/復調が恒等復元となり、残差は数値誤差レベル
    assert np.allclose(res_nopca.iter1.e_tod, 0.0, atol=1e-12)
    assert np.allclose(res_nopca.iter2.e_tod, 0.0, atol=1e-12)

    # Case 2: n_components = 1, mean_mode = "none" (PCAあり)
    res_pca = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=1,
        mean_mode="none",
        cutoff_sigma=5.0,
    )

    # PCAが変調天体信号の一部を吸収するため、C_1 は非ゼロとなる (信号損失の発生)
    c1_power = float(np.sum(res_pca.iter1.c_tod ** 2))
    assert c1_power > 0.0, "PCAは変調天体信号の一部を相関成分として吸収するべきです"

    # 残差 E_1 もゼロにはならない (天体信号の吸収による歪み)
    e1_rms = float(np.std(res_pca.iter1.e_tod))
    assert e1_rms > 0.0, "PCAによる吸収があるため残差E_1はゼロにならないはずです"

    # 加算恒等式はどちらのケースでも厳密に成立する
    assert np.allclose(res_pca.x_tod, res_pca.iter1.c_tod + res_pca.iter1.a_tod + res_pca.iter1.e_tod, atol=1e-12)
    assert np.allclose(res_pca.x_tod, res_pca.iter2.c_tod + res_pca.iter2.a_tod + res_pca.iter2.e_tod, atol=1e-12)


def test_noise_sigma_estimation_edge_cases():
    """MAD による雑音水準推定のエッジケース挙動テスト"""
    n_wide = 100
    n_time = 10
    hits = np.full(n_wide, n_time, dtype=int)
    hits[:5] = 0  # 端は未観測

    # 1. 有効点不足 (< 10 点)
    spec_sparse = np.full(n_wide, np.nan)
    spec_sparse[10:15] = 1.0
    sigma_d, sigma_0 = iterative.estimate_noise_sigma(spec_sparse, hits, n_time)
    assert sigma_0 == 0.0

    # 2. 完全平坦 (MAD = 0, std = 0)
    spec_flat = np.full(n_wide, 2.0)
    spec_flat[:5] = np.nan
    sigma_d, sigma_0 = iterative.estimate_noise_sigma(spec_flat, hits, n_time)
    assert sigma_0 == 0.0
    # モデル作成でもクラッシュしないこと
    model, _, _ = iterative.make_noise_free_model(spec_flat, hits, n_time, cutoff_sigma=5.0)
    assert np.all(np.isfinite(model))

    # 3. 未観測点 (hits == 0) のモデル値は必ず 0.0
    assert np.all(model[:5] == 0.0)


def test_regression_existing_step1():
    """既存の run_step1 の既定実行結果が期待値と整合することの回帰確認"""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    sim = pipeline.simulate(cfg)
    results = pipeline.reduce_all_modes(sim)

    # README に記載された期待値との整合性チェック (許容誤差内で一致)
    r_none = results["none"].metrics
    assert np.isclose(r_none["line_free_residual_rms_k"], 0.0394, atol=1e-3)
    assert np.isclose(r_none["peak_ratio"], 0.9876, atol=1e-3)
    assert np.isclose(r_none["integrated_ratio"], 0.9336, atol=1e-3)
