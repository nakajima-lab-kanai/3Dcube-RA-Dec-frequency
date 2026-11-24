"""既存 3 経路の基準結果の回帰テスト

キューブ入力の汎用化(observation.py の追加、pipeline.simulate のラッパー化)の前、
コミット 91c0c9c の時点で採取した値を固定してある。3 経路は設定も解析仕様も別物なので、
それぞれ独立に固定する(統合はしない)。

  1. run_step1.py           : config/default.yaml の基本解析
  2. run_fig6.py            : config/fig6_taniguchi2020.yaml(150 GHz 想定)の 2 回推定
  3. run_taniguchi2020_fig6 : 論文 Fig. 6 の図情報に基づく再現(無次元強度。K ではない)

許容誤差 rtol = 1e-9 の理由:
  同じ NumPy / BLAS ならビット単位で再現する(変更前後の全配列の一致は別途確認済み)。
  環境が変わると SVD や総和の丸めが 1e-13 程度変わりうるので、それは許し、
  処理の意味が変わる変更(桁違いに大きい差になる)だけを検出する幅にしてある。
  閾値処理(sigma-cutoff)で残るチャンネル数は整数として厳密に比べる。

注意: これは「変更前と同じ結果が出る」ことの確認であって、結果が研究上正しいことの
保証ではない。論文図との既知の不一致(README 参照)もそのまま含んでいる。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from fmlo_sim import evaluation, iterative, pipeline
from fmlo_sim import taniguchi2020 as tg
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config

HERE = Path(__file__).resolve().parent.parent
RTOL = 1e-9

STEP1 = {
    "observed_sum": 32182.639694203855,
    "observed_abs_sum": 2725653.8862419524,
    "none": (0.039417264908649, 0.9875813454113372, 0.9336082528746424),
    "channel_mean": (0.04290952465150944, 0.8309393884984699, 0.4326507586163704),
    "no_pca": (0.03717621889493397, 1.000488794012509, 0.9838054889092436),
    # 2 回推定(K=1, mean_mode=none, cutoff 5σ): (残差RMS, ピーク比, 積分強度比, モデル最大値, 非ゼロ数)
    "two1": (0.039417264908649, 0.9875813454113372, 0.9336082528746424, 1.0824040068185234, 77),
    "two2": (0.0371688386914264, 1.000447740702601, 0.9803239731747141, 1.0952204774362848, 77),
}
FIG6_150GHZ = {
    "observed_sum": 162738.4699186862,
    "observed_abs_sum": 2824286.2446766207,
    "none": (0.07862415074876722, 0.9921914190051593, 0.9444982555000265),
    "channel_mean": (0.1208460689814754, 0.8354036642410875, 0.4410576405942107),
    "no_pca": (0.03625878222428151, 1.007295979063802, 0.9995315203096566),
    "two1": (0.07862415074876722, 0.9921914190051593, 0.9444982555000265, 4.988891867438463, 107),
    "two2": (0.036363517407771266, 1.0067198585355188, 0.9975629924560682, 5.060788025943109, 113),
}


@pytest.mark.parametrize("config_path, expected", [
    (DEFAULT_CONFIG_PATH, STEP1),
    (HERE / "config" / "fig6_taniguchi2020.yaml", FIG6_150GHZ),
], ids=["run_step1_default", "run_fig6_150ghz"])
def test_basic_and_two_step_baseline(config_path, expected):
    cfg = load_config(config_path)
    sim = pipeline.simulate(cfg)
    assert np.isclose(sim.observed.sum(), expected["observed_sum"], rtol=RTOL, atol=0)
    assert np.isclose(np.abs(sim.observed).sum(), expected["observed_abs_sum"], rtol=RTOL, atol=0)

    results = pipeline.reduce_all_modes(sim)
    for label in ("none", "channel_mean", "no_pca"):
        m = results[label].metrics
        got = (m["line_free_residual_rms_k"], m["peak_ratio"], m["integrated_ratio"])
        assert np.allclose(got, expected[label], rtol=RTOL, atol=0), label

    grid = sim.grid
    two = iterative.run_two_step_estimation(
        sim.observed, sim.shifts_ch, grid.start0, grid.n_wide,
        cfg.removal.n_components, cfg.removal.mean_mode, 5.0,
    )
    for key, it in (("two1", two.iter1), ("two2", two.iter2)):
        m = evaluation.evaluate(
            it.restored_spectrum, sim.truth_k, it.hits, cfg.n_time, grid.freq_ghz,
            cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
            cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
        )
        got = (m["line_free_residual_rms_k"], m["peak_ratio"], m["integrated_ratio"], it.model_spectrum.max())
        assert np.allclose(got, expected[key][:4], rtol=RTOL, atol=0), key
        assert int(np.count_nonzero(it.model_spectrum)) == expected[key][4], key


# --- 論文 Fig. 6 再現(独立した基準事例) -----------------------------------------

# 各パネルの (総和, 絶対値の総和)。単位は図の無次元強度(E^nc の標準偏差 ≈ 1)で、K ではない
TANI_PANELS = {
    1: {"x_in": (300678.50020683015, 329409.71532210475),
        "t_cor": (300678.5002068301, 301172.37543154124),
        "t_ast": (9889.6847459928, 9889.6847459928),
        "e_nc": (-9889.684745992796, 173662.49186505863)},
    2: {"x_in": (290788.8154608373, 319538.07504168013),
        "t_cor": (290788.81546083733, 290991.47035865823),
        "t_ast": (17079.542667234597, 17079.542667234597),
        "e_nc": (-7189.85792124179, 172287.8447514928)},
}
TANI_MODEL = {1: (3.591862212011555, 42), 2: (4.870026317237146, 61)}   # (モデル最大値, 非ゼロ数)
TANI_E_NC_STD = {1: 1.0633401793572401, 2: 1.053449354318878}
TANI_VARIATION_2 = 0.3327067244275742
PROFILE_SHA256 = "c4e4d82b69e18ac4fe3dc380197cc33c96d4d81a5ac88211e30f1dcf5718ccb3"


def test_taniguchi_fig6_inputs_unchanged():
    """設定(図への当てはめ値)とデジタイズ済みプロファイルが変わっていない"""
    cfg = tg.load_fig6_config(HERE / "config" / "taniguchi2020_fig6.yaml")
    d = cfg.to_dict()
    assert d["seed"] == 2020
    assert d["spectrometer"]["n_channels"] == 2048 and d["spectrometer"]["n_time"] == 100
    assert d["fm"] == {"step_ch": 23, "n_steps": 24, "phase": 36}
    assert d["line"] == {"amplitude": 5.0, "center_ch": 1321.2, "fwhm_ch": 36.52}
    assert d["noise"] == {"sigma": 1.06}
    r = d["reduction"]
    assert r["n_components"] == [4, 1] and r["centering"] is True
    assert r["cutoff_sigma"] == 5.0 and r["two_sided_cutoff"] is False
    assert r["n_iterations"] == 2 and r["convergence_eps"] is None

    table = np.loadtxt(HERE / "data" / "taniguchi2020_fig6_correlated_profile.csv",
                       delimiter=",", comments="#", encoding="utf-8")
    assert table.shape == (100, 2)
    # テキスト → float64 の変換結果のハッシュ(改行コードや注釈行の違いには影響されない)
    assert hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest() == PROFILE_SHA256


def test_taniguchi_fig6_modulation_pattern():
    """変調パターン: 0..24,24..0(両端を 2 サンプル続ける、周期 50)× 23 ch、位相 36"""
    cfg = tg.load_fig6_config(HERE / "config" / "taniguchi2020_fig6.yaml")
    xi = tg.zigzag_fm_channels(100, cfg.fm.step_ch, cfg.fm.n_steps, cfg.fm.phase)
    base = np.concatenate([np.arange(25), np.arange(24, -1, -1)]) * 23
    assert np.array_equal(xi, np.tile(base, 3)[36:136])


def test_taniguchi_fig6_numeric_baseline():
    cfg = tg.load_fig6_config(HERE / "config" / "taniguchi2020_fig6.yaml")
    _, result = tg.run_fig6(cfg)
    assert len(result.steps) == 2
    assert [s.n_components for s in result.steps] == [4, 1]
    for i, step in enumerate(result.steps, start=1):
        for name, (total, abs_total) in TANI_PANELS[i].items():
            panel = getattr(step, name)
            # 総和は打ち消しがあるので、絶対値の総和を尺度にした絶対許容誤差で比べる
            assert abs(panel.sum() - total) <= RTOL * abs_total, (i, name)
            assert np.isclose(np.abs(panel).sum(), abs_total, rtol=RTOL, atol=0), (i, name)
        assert np.isclose(step.model.max(), TANI_MODEL[i][0], rtol=RTOL, atol=0)
        assert int(np.count_nonzero(step.model)) == TANI_MODEL[i][1]
        assert np.isclose(step.e_nc.std(), TANI_E_NC_STD[i], rtol=RTOL, atol=0)
    assert result.steps[0].variation is None or np.isnan(result.steps[0].variation)
    assert np.isclose(result.steps[1].variation, TANI_VARIATION_2, rtol=RTOL, atol=0)
