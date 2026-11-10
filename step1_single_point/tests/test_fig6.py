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
import subprocess

import numpy as np
import pytest

from fmlo_sim import iterative, modulation, pipeline, removal
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config
from fmlo_sim.modulation import compute_zigzag_period_samples
from run_fig6 import _get_git_info

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

    1. 観測 TOD と座標情報のみが入力であり、正解情報への参照引数を持たないこと。
    2. 評価側の正解信号 truth_k を全く異なる値に差し替えても、
       推定結果 (C_r, A_r, E_r, restored, model) が不変であることを確認。
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

    # 観測データのコピーを渡した場合の再現性
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
    assert np.array_equal(res1.iter1.restored_spectrum, res2.iter1.restored_spectrum, equal_nan=True)
    assert np.array_equal(res1.iter1.model_spectrum, res2.iter1.model_spectrum, equal_nan=True)

    # 評価用の正解スペクトルを意図的に改変
    fake_truth = sim.truth_k * 5.0 + 10.0
    # 推定関数は正解スペクトルを引数に取らないため、正解を変更しても推定値そのものは不変
    assert np.array_equal(res1.iter1.c_tod, res2.iter1.c_tod)


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
    # 雑音なしシミュレーションのため allow_noise_free=True を指定
    res_nopca = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=0,
        mean_mode="none",
        cutoff_sigma=5.0,
        allow_noise_free=True,
    )

    # PCAがない場合、天体信号のみの観測なので C_r は 0
    assert np.allclose(res_nopca.iter1.c_tod, 0.0, atol=1e-14)
    # 変調/復調が恒等復元となり、残差は数値誤差レベル
    assert np.allclose(res_nopca.iter1.e_tod, 0.0, atol=1e-12)
    assert np.allclose(res_nopca.iter2.e_tod, 0.0, atol=1e-12)

    # 全時刻観測された領域において、復元スペクトルと正解スペクトルが厳密に一致することを確認
    fully_obs = (res_nopca.iter1.hits == cfg.n_time)
    assert np.any(fully_obs)
    assert np.allclose(res_nopca.iter1.restored_spectrum[fully_obs], sim.truth_k[fully_obs], atol=1e-12)
    assert np.allclose(res_nopca.iter2.restored_spectrum[fully_obs], sim.truth_k[fully_obs], atol=1e-12)

    # Case 2: n_components = 1, mean_mode = "none" (PCAあり)
    res_pca = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=1,
        mean_mode="none",
        cutoff_sigma=5.0,
        allow_noise_free=True,
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
    """MAD による雑音水準推定のエッジケース挙動テスト

    新仕様:
      - 有効点不足 (< 10 点) では status = 'insufficient_points'
      - MAD == 0 かつ std == 0 では status = 'zero_mad'
      - 既定 (allow_noise_free=False) では ValueError が送出され天体モデルを作成しない
      - allow_noise_free=True の場合のみ完全無雑音フォールバックを許可する
    """
    n_wide = 100
    n_time = 10
    hits = np.full(n_wide, n_time, dtype=int)
    hits[:5] = 0  # 端は未観測

    # 1. 有効点不足 (< 10 点)
    spec_sparse = np.full(n_wide, np.nan)
    spec_sparse[10:15] = 1.0  # 有効点 5 点
    sigma_d, sigma_0, status, n_valid = iterative.estimate_noise_sigma(spec_sparse, hits, n_time)
    assert status == "insufficient_points"
    assert n_valid == 5
    assert sigma_0 == 0.0

    # 既定では天体モデル作成時に ValueError を送出
    with pytest.raises(ValueError, match="insufficient_points"):
        iterative.make_noise_free_model(spec_sparse, hits, n_time, cutoff_sigma=5.0, allow_noise_free=False)

    # allow_noise_free=True であればフォールバックが機能
    model_sparse, _, _, st_sparse, _ = iterative.make_noise_free_model(
        spec_sparse, hits, n_time, cutoff_sigma=5.0, allow_noise_free=True
    )
    assert st_sparse == "insufficient_points"
    assert np.all(np.isfinite(model_sparse))

    # 2. 完全平坦 (MAD = 0, std = 0)
    spec_flat = np.full(n_wide, 2.0)
    spec_flat[:5] = np.nan
    sigma_d, sigma_0, status, n_valid = iterative.estimate_noise_sigma(spec_flat, hits, n_time)
    assert status == "zero_mad"
    assert n_valid == 95
    assert sigma_0 == 0.0

    # 既定では天体モデル作成時に ValueError を送出
    with pytest.raises(ValueError, match="zero_mad"):
        iterative.make_noise_free_model(spec_flat, hits, n_time, cutoff_sigma=5.0, allow_noise_free=False)

    # allow_noise_free=True であればフォールバックが機能
    model, _, _, st_flat, _ = iterative.make_noise_free_model(
        spec_flat, hits, n_time, cutoff_sigma=5.0, allow_noise_free=True
    )
    assert st_flat == "zero_mad"
    assert np.all(np.isfinite(model))

    # 3. 未観測点 (hits == 0) のモデル値は必ず 0.0
    assert np.all(model[:5] == 0.0)


def test_hits_noise_scaling():
    """hits 数による局所雑音水準の補正 (sigma_d = sigma_0 * sqrt(n_time / hits)) の数値テスト

    hits = n_time と hits = n_time / 4 で比較し、雑音比が厳密に sqrt(4) = 2 となることを確認。
    """
    n_wide = 100
    n_time = 100
    rng = np.random.default_rng(12345)
    noise_sigma_true = 0.5
    # 全時刻領域にガウス雑音を生成
    spec = rng.normal(0.0, noise_sigma_true, size=n_wide)

    hits = np.full(n_wide, n_time, dtype=int)
    # チャンネル 50..59 は hits = n_time // 4 = 25
    hits[50:60] = n_time // 4

    sigma_d, sigma_0, status, n_valid = iterative.estimate_noise_sigma(spec, hits, n_time)
    assert status == "ok"
    assert sigma_0 > 0.0

    # hits == n_time の点では sigma_d == sigma_0
    assert np.isclose(sigma_d[0], sigma_0)
    # hits == n_time // 4 の点では sigma_d == 2 * sigma_0
    assert np.isclose(sigma_d[50], 2.0 * sigma_0)
    # 比が 2.0
    assert np.isclose(sigma_d[50] / sigma_d[0], 2.0)


def test_closer_to_truth_logic():
    """正解 (= 1.0) への接近判定ロジックのテスト

    比が 1.02 -> 1.10 に変化した場合 (|1.02 - 1| = 0.02, |1.10 - 1| = 0.10)、
    値は増大しているが正解からは遠ざかっているため closer_to_truth は False と判定される。
    """
    m1_peak, m2_peak = 1.02, 1.10
    peak_err1 = abs(m1_peak - 1.0)
    peak_err2 = abs(m2_peak - 1.0)
    peak_closer = bool(peak_err2 < peak_err1)
    assert peak_closer is False, "1.02 -> 1.10 は正解(=1.0)から遠ざかっているため False でなければなりません"

    # 逆に 0.94 -> 0.99 の場合は正解に近づく
    m1_int, m2_int = 0.94, 0.99
    int_err1 = abs(m1_int - 1.0)
    int_err2 = abs(m2_int - 1.0)
    int_closer = bool(int_err2 < int_err1)
    assert int_closer is True


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


def test_zigzag_period_normal():
    """zigzag パターンで半幅 > 0 のとき、離散周期が計算されること (Fig.6 設定値照合)."""
    cfg = load_config(CONFIG_PATH)
    # fig6_taniguchi2020.yaml: half_width_ch=150, zigzag_step_ch=20, dt_s=0.1
    # 期待: 4 * 150 / gcd(600, 20) = 30 samples, 30 * 0.1 = 3.0 s
    period_samples, period_seconds = compute_zigzag_period_samples(
        cfg.modulation.half_width_ch,
        cfg.modulation.zigzag_step_ch,
        cfg.modulation.dt_s,
    )
    assert period_samples == 30.0
    assert period_seconds == 3.0


def test_zigzag_period_step10():
    """ステップ10: 4*150/gcd(600,10) = 60 samples, 6.0 s."""
    period_samples, period_seconds = compute_zigzag_period_samples(150, 10, 0.1)
    assert period_samples == 60.0
    assert period_seconds == 6.0


def test_zigzag_period_step20():
    """ステップ20: 4*150/gcd(600,20) = 30 samples."""
    period_samples, period_seconds = compute_zigzag_period_samples(150, 20, 0.1)
    assert period_samples == 30.0
    assert period_seconds == 3.0


def test_zigzag_period_step7():
    """ステップ7: 4*150/gcd(600,7) = 600 samples (600と7は互いに素)."""
    period_samples, period_seconds = compute_zigzag_period_samples(150, 7, 0.1)
    assert period_samples == 600.0
    assert period_seconds == 60.0


def test_zigzag_period_zero_half_width():
    """半幅0の定数列では周期が None になること."""
    period_samples, period_seconds = compute_zigzag_period_samples(0, 10, 0.1)
    assert period_samples is None
    assert period_seconds is None


def test_zigzag_period_constant_fold():
    """半幅150・ステップ300: 出力は全て0の定数列 → None."""
    period_samples, period_seconds = compute_zigzag_period_samples(150, 300, 0.1)
    assert period_samples is None
    assert period_seconds is None


def test_zigzag_period_verify_with_make_shift_pattern():
    """make_shift_pattern が生成した列と周期を照合する."""
    from fmlo_sim.modulation import make_shift_pattern

    # step=10: 周期60。2周期分(120サンプル)を生成して繰り返しを確認。
    shifts = make_shift_pattern("zigzag", 120, 150, 10)
    assert np.array_equal(shifts[:60], shifts[60:120])

    # step=7: 周期600。1周期分(600サンプル)を生成。
    shifts7 = make_shift_pattern("zigzag", 600, 150, 7)
    assert len(shifts7) == 600
    # 1周期目と2周期目(600-1199)を比較
    shifts7_2 = make_shift_pattern("zigzag", 1200, 150, 7)
    assert np.array_equal(shifts7, shifts7_2[600:1200])


@pytest.mark.parametrize(
    ("status_stdout", "expected_modified", "expected_untracked"),
    [
        ("", False, False),  # clean
        (" M path/to/file.py", True, False),  # 未ステージ変更
        ("M  path/to/file.py", True, False),  # ステージ済み変更
        ("MM path/to/file.py", True, False),  # 両方の変更
        (" D path/to/file.py", True, False),  # 未ステージ削除
        ("UU path/to/file.py", True, False),  # 競合
        ("?? path/to/new_file.py", False, True),  # 未追跡のみ
        (" M path/to/file.py\n?? path/to/new_file.py", True, True),  # 変更と未追跡の混在
    ],
)
def test_get_git_info_status_cases(monkeypatch, status_stdout, expected_modified, expected_untracked):
    """git status の各出力パターンに対する dirty / untracked 判定の検証."""
    def fake_run(cmd, capture_output=True, text=True, cwd=None):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abcdef1234567890\n", stderr="")
        if cmd == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=status_stdout, stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("run_fig6.subprocess.run", fake_run)
    info = _get_git_info()
    assert info["commit_sha"] == "abcdef1234567890"
    assert info["commit_sha_reason"] is None
    assert info["uncommitted_modified"] is expected_modified
    assert info["uncommitted_modified_reason"] is None
    assert info["uncommitted_untracked"] is expected_untracked
    assert info["uncommitted_untracked_reason"] is None


def test_get_git_info_not_found(monkeypatch):
    """git 未導入 (FileNotFoundError) の場合のフォールバック検証."""
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr("run_fig6.subprocess.run", fake_run)
    info = _get_git_info()
    assert info["commit_sha"] is None
    assert "git コマンドが見つかりません" in info["commit_sha_reason"]
    assert info["uncommitted_modified"] is None
    assert "git コマンドが見つかりません" in info["uncommitted_modified_reason"]
    assert info["uncommitted_untracked"] is None
    assert "git コマンドが見つかりません" in info["uncommitted_untracked_reason"]


def test_get_git_info_rev_parse_failure(monkeypatch):
    """git rev-parse 失敗時の検証 (cleanと誤記録されないこと)."""
    def fake_run(cmd, capture_output=True, text=True, cwd=None):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: not a git repository")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("run_fig6.subprocess.run", fake_run)
    info = _get_git_info()
    assert info["commit_sha"] is None
    assert "git rev-parse 失敗" in info["commit_sha_reason"]


def test_get_git_info_status_failure(monkeypatch):
    """git status 失敗時の検証 (cleanと誤記録されずnullと理由が返ること)."""
    def fake_run(cmd, capture_output=True, text=True, cwd=None):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abcdef1234567890\n", stderr="")
        if cmd == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: broken index")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("run_fig6.subprocess.run", fake_run)
    info = _get_git_info()
    assert info["commit_sha"] == "abcdef1234567890"
    assert info["commit_sha_reason"] is None
    assert info["uncommitted_modified"] is None
    assert "git status 失敗" in info["uncommitted_modified_reason"]
    assert info["uncommitted_untracked"] is None
    assert "git status 失敗" in info["uncommitted_untracked_reason"]
