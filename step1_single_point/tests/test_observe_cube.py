"""外部入力キューブの固定点模擬観測(observation / obsio / analysis)のテスト

比較の許容誤差とその理由:
  - 保存 → 再読込: 配列は float64 のまま保存するので丸めが起きない。厳密一致で比べる。
  - 直接解析と再読込後の解析: 入力がビット単位で同じで処理も同じなので、厳密一致で比べる。
  - 成分の和: 生成時と同じ順序で足し直すので厳密一致で比べる。
  - 雑音なし・除去なしの復元: 復調は「同じ値を hits 回足して hits で割る」計算になる。
    float32 由来の値(仮数 24 bit)は 600 回程度の和が float64 で正確に表せるので厳密一致。
    float64 の値では和の丸めが 1 回あたり最大 0.5 ulp 生じるので、相対誤差
    n_time × eps(eps = 2.2e-16)を上限とする。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import run_analyze_observation
import run_observe_cube
from fmlo_sim import analysis, cube as cube_mod, observation, obsio, pipeline
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config
from fmlo_sim.observation import CubeInputError

EPS = np.finfo(np.float64).eps

# 小さな分光器(64 ch、変調 ±6 ch、20 サンプル)。チャンネル幅は 1 MHz
N_CH, HALF, N_TIME, DNU_MHZ, CENTER = 64, 6, 20, 1.0, 100.0


def small_setup_dict(thermal=True, atmosphere=True, ra=0.0, dec=0.0, **mod):
    return {
        "seed": 7,
        "pointing": {"ra_arcsec": ra, "dec_arcsec": dec},
        "spectrometer": {"n_channels": N_CH, "channel_width_mhz": DNU_MHZ, "center_freq_ghz": CENTER},
        "modulation": {"pattern": "zigzag", "dt_s": 0.1, "half_width_ch": HALF, "zigzag_step_ch": 2, **mod},
        "observation": {"duration_s": N_TIME * 0.1},
        "noise": {
            "thermal": {"enabled": thermal, "tsys_k": 100.0},
            "atmosphere": {"enabled": atmosphere, "model": "laminar", "alpha": 1.0,
                           "rms_k": 0.5, "offset_k": 2.0, "slope_frac": 0.1},
        },
    }


def make_setup(tmp_path, name="setup.yaml", overrides=None, **kwargs):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(small_setup_dict(**kwargs)), encoding="utf-8")
    setup, _ = observation.load_observing_setup(path, overrides)
    return setup


def small_axes(extra_ch=3):
    """分光器の格子に重なる周波数軸(変調幅 + extra_ch の余白)と 3 × 2 の空間軸"""
    start0 = HALF + extra_ch
    n_wide = N_CH + 2 * start0
    freq = CENTER + (np.arange(n_wide) - start0 - (N_CH - 1) / 2.0) * DNU_MHZ * 1e-3
    return np.array([-10.0, 0.0, 10.0]), np.array([-5.0, 5.0]), freq, start0


def small_cube(dtype=np.float64, extra_ch=3):
    """位置ごとに異なる既知のスペクトルを持つ非ガウシアンの小さなキューブ

    画素 (i, j) のスペクトル = 階段 + のこぎり波 + 画素ごとに違う位置の矩形(負値も含む)。
    """
    ra, dec, freq, start0 = small_axes(extra_ch)
    k = np.arange(freq.size)
    data = np.empty((ra.size, dec.size, freq.size))
    for i in range(ra.size):
        for j in range(dec.size):
            spec = 0.25 * (i + 1) * (k % 7) - 0.5 * j + 0.01 * k
            spec[10 + 5 * i + 3 * j: 20 + 5 * i + 3 * j] += 3.0 + i - 2.0 * j
            data[i, j] = spec
    return observation.SkyCube(data.astype(dtype), ra, dec, freq, identity={"kind": "array", "name": "test"})


# ---------------------------------------------------------------------------
# 1. 内部生成キューブ: 直接与えた場合 = 保存・再読込して与えた場合
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_sim():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    return cfg, pipeline.simulate(cfg)


def assert_same_observation(a, b):
    assert np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.signal_tod, b.signal_tod)
    assert np.array_equal(a.truth_k, b.truth_k)
    assert np.array_equal(a.shifts_ch, b.shifts_ch)
    assert np.array_equal(a.time_s, b.time_s)
    assert np.array_equal(a.grid.freq_ghz, b.grid.freq_ghz)
    assert (a.grid.start0, a.grid.n_channels) == (b.grid.start0, b.grid.n_channels)
    assert list(a.noise.components) == list(b.noise.components)
    for name in a.noise.components:
        assert np.array_equal(a.noise.components[name], b.noise.components[name])


def test_internal_cube_direct_equals_saved_and_reloaded(default_sim, tmp_path):
    cfg, sim = default_sim
    cube_path = tmp_path / "cube.npz"
    cube_mod.save_cube(cube_path, sim.cube, sim.ra_arcsec, sim.dec_arcsec, sim.grid.freq_ghz, cfg)

    sky = observation.load_sky_cube(cube_path)
    assert sky.data.dtype == sim.cube.dtype and np.array_equal(sky.data, sim.cube)
    obs = observation.observe(sky, pipeline.internal_observing_setup(cfg))

    assert_same_observation(sim, obs)
    assert obs.pixel == sim.pixel
    # 周波数格子はキューブの軸から求めたもので、内部生成時の格子と一致する
    grid = cube_mod.make_frequency_grid(cfg)
    assert obs.grid.start0 == grid.start0 and obs.grid.n_wide == grid.n_wide
    assert obs.grid.channel_width_ghz == grid.channel_width_ghz


def test_simulate_wrapper_keeps_public_fields(default_sim):
    cfg, sim = default_sim
    assert isinstance(sim, pipeline.Simulation)
    assert sim.cube.shape == (41, 41, sim.grid.n_wide) and sim.cube.dtype == np.float32
    assert sim.observed.shape == (cfg.n_time, cfg.spectrometer.n_channels)
    assert sim.pixel == (20, 20)


# ---------------------------------------------------------------------------
# 2. 非ガウシアンの小さなキューブ: 指定画素の正しいスペクトルが観測される
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ra, dec, pixel", [
    (-10.0, -5.0, (0, 0)), (0.0, 5.0, (1, 1)), (10.0, -5.0, (2, 0)),
    (3.0, 1.0, (1, 1)),       # 最近傍画素
    (-14.9, 9.9, (0, 1)),     # 端の画素の内側(縁は ±半画素まで)
])
def test_nongaussian_cube_pixel_spectrum(tmp_path, ra, dec, pixel):
    sky = small_cube()
    setup = make_setup(tmp_path, thermal=False, atmosphere=False, ra=ra, dec=dec)
    obs = observation.observe(sky, setup)
    assert obs.pixel == pixel
    expected = sky.data[pixel[0], pixel[1]]
    assert np.array_equal(obs.truth_k, expected)
    # 各時刻の窓 = キューブのスペクトルを start0 + shift から n_channels 切り出したもの
    for t in range(setup.n_time):
        s = obs.grid.start0 + obs.shifts_ch[t]
        assert np.array_equal(obs.signal_tod[t], expected[s:s + N_CH])
        assert np.array_equal(obs.grid.freq_ghz[s:s + N_CH], sky.freq_ghz[s:s + N_CH])
    assert np.array_equal(obs.observed, obs.signal_tod)   # 雑音なし
    assert obs.shifts_ch.min() == -HALF and obs.shifts_ch.max() == HALF


# ---------------------------------------------------------------------------
# 3. 雑音なし・除去なし: 観測された周波数の復元値 = 入力、未観測は NaN(既存仕様)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noiseless_no_removal_restores_input(tmp_path, dtype):
    sky = small_cube(dtype=dtype)
    setup = make_setup(tmp_path, thermal=False, atmosphere=False, ra=0.0, dec=5.0)
    obs = observation.observe(sky, setup)
    red = pipeline.reduce_observed(obs.observed, obs.shifts_ch, obs.grid.start0, obs.grid.n_wide, 0, "none")

    truth = sky.data[1, 1].astype(np.float64)
    seen = red.hits > 0
    # 余白(extra_ch = 3)の分だけ両端に未観測の周波数が残る
    assert np.array_equal(np.flatnonzero(~seen), np.r_[0:3, truth.size - 3:truth.size])
    assert np.all(np.isnan(red.restored_k[~seen])) and np.all(red.hits[~seen] == 0)
    assert np.all(np.isfinite(red.restored_k[seen]))
    if dtype is np.float32:
        assert np.array_equal(red.restored_k[seen], truth[seen])
    else:
        err = np.abs(red.restored_k[seen] - truth[seen])
        assert np.all(err <= setup.n_time * EPS * np.abs(truth[seen]))
    assert red.hits.max() == setup.n_time and red.hits.sum() == setup.n_time * N_CH


# ---------------------------------------------------------------------------
# 4. 雑音の 4 通り: 成分の和 = 観測時系列、乱数列は既存方式のまま
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thermal, atmosphere, names", [
    (False, False, []), (True, False, ["thermal"]),
    (False, True, ["atmosphere"]), (True, True, ["thermal", "atmosphere"]),
])
def test_noise_combinations_sum_to_observed(tmp_path, thermal, atmosphere, names):
    sky = small_cube()
    obs = observation.observe(sky, make_setup(tmp_path, thermal=thermal, atmosphere=atmosphere))
    assert list(obs.noise.components) == names
    total = np.zeros(obs.observed.shape)
    for tod in obs.noise.components.values():
        assert tod.shape == (N_TIME, N_CH) and np.all(np.isfinite(tod))
        total += tod
    assert np.array_equal(obs.observed, obs.signal_tod + total)
    if not names:
        assert np.array_equal(obs.observed, obs.signal_tod)

    # 片方の成分を切っても、もう片方の乱数列は変わらない(成分別の乱数生成器)
    both = observation.observe(sky, make_setup(tmp_path, thermal=True, atmosphere=True))
    for name in names:
        assert np.array_equal(obs.noise.components[name], both.noise.components[name])


def test_rng_scheme_unchanged(default_sim):
    """熱雑音は spawn_rngs(seed)['thermal'] の正規乱数そのもの(既存の生成方式)"""
    from fmlo_sim import noise
    from fmlo_sim.config import spawn_rngs
    cfg, sim = default_sim
    rng = spawn_rngs(cfg.seed)["thermal"]
    expected = noise.thermal_noise(cfg.n_time, cfg.spectrometer.n_channels, cfg.noise.thermal.tsys_k,
                                   cfg.spectrometer.channel_width_mhz * 1e6, cfg.modulation.dt_s, rng)
    assert np.array_equal(sim.noise.components["thermal"], expected)


# ---------------------------------------------------------------------------
# 5. 保存 → 再読込 → 解析 = 直接解析(NaN の位置・ヒット数・軸も比べる)
# ---------------------------------------------------------------------------

def test_saved_observation_roundtrip_and_analysis(default_sim, tmp_path):
    cfg, sim = default_sim
    sky = pipeline.make_internal_sky_cube(cfg)
    obs = observation.observe(sky, pipeline.internal_observing_setup(cfg))
    path = tmp_path / "obs.npz"
    meta = obsio.save_observation(path, obs, sky, provenance={"note": "test"})
    data, truth = obsio.load_observation(path)

    # 保存前後: float64 無損失なので厳密一致。型も変わらない
    assert data.observed.dtype == np.float64 and np.array_equal(data.observed, obs.observed)
    assert np.array_equal(data.shifts_ch, obs.shifts_ch) and np.array_equal(data.time_s, obs.time_s)
    assert np.array_equal(data.freq_wide_ghz, obs.grid.freq_ghz)
    assert np.array_equal(truth.truth_k, obs.truth_k) and np.array_equal(truth.signal_tod, obs.signal_tod)
    assert list(truth.noise) == list(obs.noise.components)
    for name, tod in truth.noise.items():
        assert np.array_equal(tod, obs.noise.components[name])
    assert (data.grid.start0, data.grid.n_channels, data.grid.channel_width_ghz, data.grid.center_freq_ghz) == \
        (obs.grid.start0, obs.grid.n_channels, obs.grid.channel_width_ghz, obs.grid.center_freq_ghz)
    assert analysis.check_composition(data, truth) == {"max_abs_composition_err_k": 0.0, "is_exact": True}

    # 記録内容
    assert meta["precision"]["float_arrays"] == "float64"
    assert meta["seed"] == cfg.seed and meta["setup"] == obs.setup.to_dict()
    assert meta["cube"]["content_sha256"] == sky.content_sha256()
    assert meta["cube"]["units"] == cube_mod.CUBE_UNITS and meta["cube"]["shape"] == list(sky.data.shape)
    assert meta["pointing"]["pixel_index"] == [20, 20]
    assert data.meta == json.loads(json.dumps(meta))
    assert observation.setup_from_dict(data.meta["setup"]).to_dict() == obs.setup.to_dict()

    # 直接解析(既存の一括経路)と、再読込したデータの解析
    direct = pipeline.reduce_all_modes(sim)
    result = analysis.analyze(data, cfg.removal.n_components, cfg.removal.mean_mode,
                              two_step=True, cutoff_sigma=5.0)
    line_def = analysis.line_definition_from_source(data.meta, cfg.evaluation.line_free_n_fwhm)
    metrics = analysis.evaluate(result, data, truth, line_def)
    assert list(result.basic) == list(direct)
    for label, r in direct.items():
        b = result.basic[label]
        assert np.array_equal(b.hits, r.hits)
        assert np.array_equal(np.isnan(b.restored_k), np.isnan(r.restored_k))
        assert np.array_equal(b.restored_k, r.restored_k, equal_nan=True)
        assert metrics["basic"][label]["line_metrics"] == r.metrics
    assert np.isnan(direct["none"].restored_k).sum() == 2 * cfg.cube.margin_ch   # 未観測 = 余白

    from fmlo_sim import iterative
    two = iterative.run_two_step_estimation(sim.observed, sim.shifts_ch, sim.grid.start0, sim.grid.n_wide,
                                            cfg.removal.n_components, cfg.removal.mean_mode, 5.0)
    for a, b in ((two.iter1, result.two_step.iter1), (two.iter2, result.two_step.iter2)):
        assert np.array_equal(a.restored_spectrum, b.restored_spectrum, equal_nan=True)
        assert np.array_equal(a.model_spectrum, b.model_spectrum)
        assert np.array_equal(a.hits, b.hits)


def test_scripts_generate_then_analyze(tmp_path):
    """A(生成・保存)と B(読込・解析)を別々に実行でき、既存出力を上書きしない"""
    sky = small_cube()
    cube_path = tmp_path / "cube.npz"
    cube_mod.save_cube(cube_path, sky.data, sky.ra_arcsec, sky.dec_arcsec, sky.freq_ghz)
    cfg_path = tmp_path / "setup.yaml"
    cfg_dict = small_setup_dict(ra=10.0, dec=-5.0)
    cfg_dict["removal"] = {"n_components": 1, "mean_mode": "none"}
    cfg_dict["evaluation"] = {"line_free_n_fwhm": 2.0}
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    obs_path = tmp_path / "out" / "obs.npz"
    obs, _, meta = run_observe_cube.observe_and_save(obs_path, cube_path=cube_path, config_path=cfg_path)
    assert obs.pixel == (2, 0)
    assert meta["cube"]["identity"]["kind"] == "file" and len(meta["cube"]["identity"]["file_sha256"]) == 64
    assert meta["cube"]["identity"]["file_meta"]["config"] is None   # 外部キューブ: 生成条件なし
    git = meta["provenance"]["git"]
    assert "commit_sha" in git and (git["commit_sha"] is not None or git["commit_sha_reason"])
    assert meta["provenance"]["environment"]["packages"]["numpy"]["version"] == np.__version__
    assert meta["provenance"]["config_sections_not_used"] == ["removal", "evaluation"]
    with pytest.raises(FileExistsError):
        run_observe_cube.observe_and_save(obs_path, cube_path=cube_path, config_path=cfg_path)

    outdir = tmp_path / "ana"
    line_def = analysis.LineDefinition("windows", windows_ghz=[(99.99, 100.0)], ref_freq_ghz=100.0)
    summary = run_analyze_observation.analyze_and_save(obs_path, outdir, config_path=cfg_path, line_def=line_def)
    assert summary["composition_check"]["is_exact"] is True
    assert summary["analysis_conditions"]["truth_used_in_estimation"] is False
    assert summary["observation"]["file_sha256"] == obsio.sha256_of_file(obs_path)
    saved = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["results"] == summary["results"]
    no_pca = summary["results"]["basic"]["no_pca"]
    assert no_pca["line_metrics"]["n_line_channels"] > 0 and no_pca["residual_rms_k"] is not None
    with np.load(outdir / "analysis.npz") as f:
        assert f["restored_no_pca_k"].dtype == np.float64 and f["hits"].sum() == N_TIME * N_CH
    with pytest.raises(FileExistsError):
        run_analyze_observation.analyze_and_save(obs_path, outdir, config_path=cfg_path, line_def=line_def)

    # 輝線領域を指定しなければ、輝線に依る評価値は「未計算」と理由が残る
    s2 = run_analyze_observation.analyze_and_save(obs_path, tmp_path / "ana2", config_path=cfg_path)
    assert s2["results"]["basic"]["none"]["line_metrics"] is None
    assert s2["results"]["basic"]["none"]["line_metrics_reason"]
    # 生成条件のないキューブで --line-from-source を使うとエラー(推測しない)
    with pytest.raises(ValueError):
        run_analyze_observation.analyze_and_save(obs_path, tmp_path / "ana3", config_path=cfg_path,
                                                 line_from_source=True)


def test_estimation_does_not_use_truth(tmp_path):
    """正解情報(正解スペクトル・天体信号・雑音成分)を壊しても推定結果は変わらない"""
    sky = small_cube()
    obs = observation.observe(sky, make_setup(tmp_path))
    path = tmp_path / "obs.npz"
    obsio.save_observation(path, obs, sky)
    data, truth = obsio.load_observation(path)
    before = analysis.analyze(data, 1, "none")

    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    for key in ("truth_k", "signal_tod", "noise_thermal", "noise_atmosphere"):
        arrays[key] = np.full_like(arrays[key], 123.0)
    path2 = tmp_path / "obs_broken_truth.npz"
    np.savez(path2, **arrays)
    data2, _ = obsio.load_observation(path2)
    after = analysis.analyze(data2, 1, "none")
    for label in before.basic:
        assert np.array_equal(before.basic[label].restored_k, after.basic[label].restored_k, equal_nan=True)
    assert not hasattr(data, "truth_k") and not hasattr(data, "signal_tod")


# ---------------------------------------------------------------------------
# 6. 入力検証
# ---------------------------------------------------------------------------

def _sky(data=None, ra=None, dec=None, freq=None):
    base = small_cube()
    return observation.SkyCube(
        base.data if data is None else data, base.ra_arcsec if ra is None else ra,
        base.dec_arcsec if dec is None else dec, base.freq_ghz if freq is None else freq,
    )


def test_rejects_shape_mismatch(tmp_path):
    setup = make_setup(tmp_path)
    base = small_cube()
    for bad in (
        _sky(ra=np.array([-10.0, 0.0])),                    # RA 軸の長さ違い
        _sky(dec=np.array([-5.0, 0.0, 5.0])),               # Dec 軸の長さ違い
        _sky(freq=base.freq_ghz[:-1]),                      # 周波数軸の長さ違い
        _sky(data=base.data[0]),                            # 2 次元
        _sky(data=base.data.transpose(2, 0, 1)),            # 軸順序違い(形が合わない)
        _sky(ra=base.ra_arcsec.reshape(3, 1)),              # 軸が 1 次元でない
    ):
        with pytest.raises(CubeInputError):
            observation.observe(bad, setup)


def test_rejects_unsupported_axes(tmp_path):
    setup = make_setup(tmp_path)
    base = small_cube()
    f = base.freq_ghz
    uneven = f.copy(); uneven[5] += 0.3e-3
    for bad in (
        _sky(ra=base.ra_arcsec[::-1].copy()),               # 降順
        _sky(ra=np.array([-10.0, 0.0, 15.0])),              # 空間軸が不等間隔
        _sky(freq=f[::-1].copy()),                          # 周波数が降順
        _sky(freq=uneven),                                  # 周波数が不等間隔
        _sky(freq=CENTER + (f - CENTER) * 2.0),             # チャンネル幅が分光器と違う
        _sky(freq=f + 0.4e-3),                              # 分光器の格子から 0.4 ch ずれ
        _sky(freq=f.astype(np.float32)),                    # 精度不足(float32)で格子に乗らない
        _sky(data=base.data.astype(complex)),               # 複素数
    ):
        with pytest.raises(CubeInputError):
            observation.observe(bad, setup)


def test_rejects_nonfinite(tmp_path):
    setup = make_setup(tmp_path)
    base = small_cube()
    for value in (np.nan, np.inf, -np.inf):
        data = base.data.copy(); data[2, 1, 40] = value     # 観測しない画素でも不可
        with pytest.raises(CubeInputError):
            observation.observe(_sky(data=data), setup)
    masked = np.ma.masked_invalid(base.data)
    with pytest.raises(CubeInputError):
        observation.observe(_sky(data=masked), setup)
    ra = base.ra_arcsec.copy(); ra[0] = np.nan
    with pytest.raises(CubeInputError):
        observation.observe(_sky(ra=ra), setup)


@pytest.mark.parametrize("ra, dec", [(15.1, 0.0), (-15.1, 0.0), (0.0, 10.1), (0.0, -10.1), (1e6, 0.0)])
def test_rejects_pointing_outside_map(tmp_path, ra, dec):
    """マップ外は端の画素へ丸めずエラー(縁は端の画素中心 ± 半画素)"""
    with pytest.raises(CubeInputError):
        observation.observe(small_cube(), make_setup(tmp_path, ra=ra, dec=dec))


def test_rejects_insufficient_frequency_band(tmp_path):
    base = small_cube(extra_ch=0)                           # ちょうど変調幅ぶんだけの帯域
    observation.observe(base, make_setup(tmp_path))         # これは通る
    for sl in (slice(1, None), slice(None, -1)):            # 片側 1 ch 足りない
        cut = observation.SkyCube(base.data[:, :, sl], base.ra_arcsec, base.dec_arcsec, base.freq_ghz[sl])
        with pytest.raises(CubeInputError):
            observation.observe(cut, make_setup(tmp_path))
    # 分光器の帯域そのものが入らない
    narrow = observation.SkyCube(base.data[:, :, 20:60], base.ra_arcsec, base.dec_arcsec, base.freq_ghz[20:60])
    with pytest.raises(CubeInputError):
        observation.observe(narrow, make_setup(tmp_path))
    # 変調幅を広げても足りなくなる
    with pytest.raises(CubeInputError):
        observation.observe(base, make_setup(tmp_path, half_width_ch=HALF + 2))   # step 2 で実際に ±8 ch まで届く


def test_rejects_bad_cube_file_and_setup(tmp_path):
    sky = small_cube()
    good = tmp_path / "cube.npz"
    cube_mod.save_cube(good, sky.data, sky.ra_arcsec, sky.dec_arcsec, sky.freq_ghz)
    with np.load(good) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays["meta"]))

    def write(name, **changes):
        path = tmp_path / name
        np.savez(path, **{**arrays, **changes})
        return path

    bad_units = dict(meta, units={**meta["units"], "freq": "Hz"})
    bad_axes = dict(meta, axes="cube[i_freq, i_dec, i_ra]")
    with pytest.raises(CubeInputError):
        observation.load_sky_cube(write("units.npz", meta=json.dumps(bad_units)))
    with pytest.raises(CubeInputError):
        observation.load_sky_cube(write("axes.npz", meta=json.dumps(bad_axes)))
    no_meta = tmp_path / "nometa.npz"
    np.savez(no_meta, **{k: v for k, v in arrays.items() if k != "meta"})
    with pytest.raises(CubeInputError):
        observation.load_sky_cube(no_meta)

    # 観測設定: pointing が null、未知のキー
    d = small_setup_dict(); d["pointing"]["ra_arcsec"] = None
    p = tmp_path / "null_pointing.yaml"; p.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError):
        observation.load_observing_setup(p)
    d = small_setup_dict(); d["beam"] = {"fwhm_arcsec": 15.0}
    p = tmp_path / "unknown.yaml"; p.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError):
        observation.load_observing_setup(p)


# ---------------------------------------------------------------------------
# 7. 入力キューブと観測時系列を変更しない
# ---------------------------------------------------------------------------

def test_inputs_not_modified(tmp_path):
    sky = small_cube()
    copies = [a.copy() for a in (sky.data, sky.ra_arcsec, sky.dec_arcsec, sky.freq_ghz)]
    obs = observation.observe(sky, make_setup(tmp_path))
    path = tmp_path / "obs.npz"
    obsio.save_observation(path, obs, sky)
    for a, c in zip((sky.data, sky.ra_arcsec, sky.dec_arcsec, sky.freq_ghz), copies):
        assert np.array_equal(a, c)

    data, truth = obsio.load_observation(path)
    observed, shifts = data.observed.copy(), data.shifts_ch.copy()
    truth_k = truth.truth_k.copy()
    result = analysis.analyze(data, 1, "none", two_step=True, cutoff_sigma=5.0)
    analysis.evaluate(result, data, truth)
    assert np.array_equal(data.observed, observed) and np.array_equal(data.shifts_ch, shifts)
    assert np.array_equal(truth.truth_k, truth_k)
    assert np.array_equal(obs.observed, observed)
