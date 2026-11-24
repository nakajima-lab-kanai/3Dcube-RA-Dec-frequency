"""天体キューブ(RA × Dec × 周波数)の固定点・1ビーム模擬観測

キューブの作り方(内部生成のガウシアン / 外部入力)と観測処理を分離する。
observe() は SkyCube と ObservingSetup だけを受け取り、時間 × 分光器チャンネルの
観測時系列を作る。キューブ(空間 × 空間 × 周波数)と観測時系列(時間 × チャンネル)は
別物なので混同しないこと。

対応する入力(これ以外は明確なエラーにする):
  - 値の単位 K、欠損を含まない有限値の実数配列、軸順序 cube[i_ra, i_dec, i_freq]
  - 空間軸: 平面上の角度オフセット [arcsec]、昇順・等間隔(1 画素だけの軸も可)
  - 周波数軸: [GHz]、昇順・等間隔。チャンネル幅は分光器と同じで、変調なしの観測窓の
    チャンネルがキューブのチャンネルにちょうど重なること
  - 固定点のサンプリング: 最近傍画素(既存の cube.extract_spectrum)

行わないこと: 周波数軸の作り直し・補間・外挿・循環シフト・ゼロ埋め、空間補間、
ビーム畳み込み、マップ外の指向位置を端の画素へ丸めること。
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import cube as cube_mod
from . import modulation, noise
from .config import (
    Config, ModulationConfig, NoiseConfig, ObservationConfig, SpectrometerConfig,
    _apply_override, _build, spawn_rngs, validate_observing,
)

# 軸が等間隔か、分光器のチャンネルに重なるかの判定に使う許容誤差(間隔に対する比)。
# float64 で 150 GHz 付近の周波数を表すと丸め誤差は約 3e-14 GHz で、チャンネル幅
# 0.5 MHz に対して 6e-11。格子を作る計算で数倍に増えても 1e-6 には遠く届かない。
# 一方 float32 の周波数軸(誤差 約 1.5e-5 GHz = 0.03 ch)はこの判定で不合格になる。
AXIS_RTOL = 1e-6

# 観測設定 YAML のうち observe() が使うセクションと、既存の設定ファイルを流用できるよう
# 置いてあっても無視するセクション
SETUP_SECTIONS = ("seed", "pointing", "spectrometer", "modulation", "observation", "noise")
IGNORED_SECTIONS = ("source", "cube", "removal", "evaluation")


class CubeInputError(ValueError):
    """入力キューブ・観測設定が対応範囲外のときのエラー"""


# ---------------------------------------------------------------------------
# 入力キューブ
# ---------------------------------------------------------------------------

@dataclass
class SkyCube:
    """天球キューブ。data[i_ra, i_dec, i_freq] [K]"""

    data: np.ndarray
    ra_arcsec: np.ndarray
    dec_arcsec: np.ndarray
    freq_ghz: np.ndarray
    identity: dict = field(default_factory=dict)   # 出どころの記録(種類・パスなど)

    def content_sha256(self) -> str:
        """配列の中身(値・軸・形・型)から計算した識別用ハッシュ"""
        h = hashlib.sha256()
        for a in (self.data, self.ra_arcsec, self.dec_arcsec, self.freq_ghz):
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
        return h.hexdigest()


def _check_axis(name, axis, n_expected, require_uniform=True):
    axis = np.asarray(axis)
    if isinstance(axis, np.ma.MaskedArray):
        raise CubeInputError(f"{name} 軸がマスク付き配列です(欠損には対応していません)")
    if axis.ndim != 1:
        raise CubeInputError(f"{name} 軸は 1 次元である必要があります(形 {axis.shape})")
    if axis.dtype.kind not in "fiu":
        raise CubeInputError(f"{name} 軸の型 {axis.dtype} には対応していません")
    if axis.size != n_expected:
        raise CubeInputError(
            f"{name} 軸の長さ {axis.size} がキューブの形と合いません(期待値 {n_expected})"
        )
    if not np.all(np.isfinite(axis)):
        raise CubeInputError(f"{name} 軸に非有限値(NaN/inf)があります")
    if axis.size >= 2:
        d = np.diff(axis.astype(float))
        if not np.all(d > 0):
            raise CubeInputError(f"{name} 軸は昇順(狭義単調増加)である必要があります")
        if require_uniform:
            step = (float(axis[-1]) - float(axis[0])) / (axis.size - 1)
            ideal = float(axis[0]) + np.arange(axis.size) * step
            if np.max(np.abs(axis - ideal)) > AXIS_RTOL * step:
                raise CubeInputError(f"{name} 軸が等間隔ではありません(不等間隔には対応していません)")


def validate_sky_cube(sky: SkyCube) -> None:
    """入力キューブが対応範囲内か検証する。範囲外なら CubeInputError"""
    data = sky.data
    if isinstance(data, np.ma.MaskedArray):
        raise CubeInputError("キューブがマスク付き配列です(欠損には対応していません)")
    if not isinstance(data, np.ndarray):
        raise CubeInputError("キューブは numpy 配列である必要があります")
    if data.ndim != 3:
        raise CubeInputError(
            f"キューブは 3 次元 (RA, Dec, 周波数) である必要があります(形 {data.shape})"
        )
    if data.dtype.kind not in "fiu":
        raise CubeInputError(f"キューブの型 {data.dtype} には対応していません(実数のみ)")
    if 0 in data.shape:
        raise CubeInputError(f"キューブに長さ 0 の軸があります(形 {data.shape})")
    _check_axis("RA", sky.ra_arcsec, data.shape[0])
    _check_axis("Dec", sky.dec_arcsec, data.shape[1])
    _check_axis("周波数", sky.freq_ghz, data.shape[2])
    if data.shape[2] < 2:
        raise CubeInputError("周波数軸は 2 チャンネル以上必要です")
    if not np.all(np.isfinite(data)):
        n_bad = int(np.count_nonzero(~np.isfinite(data)))
        raise CubeInputError(f"キューブに非有限値(NaN/inf)が {n_bad} 個あります(欠損には対応していません)")


def load_sky_cube(path) -> SkyCube:
    """cube.save_cube の保存形式(.npz)を読み込む

    必要なキー: cube, ra_arcsec, dec_arcsec, freq_ghz, meta。meta の axes と units が
    対応する仕様(cube[i_ra, i_dec, i_freq]、K / arcsec / GHz)と一致しない場合はエラー。
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in ("cube", "ra_arcsec", "dec_arcsec", "freq_ghz", "meta") if k not in f.files]
        if missing:
            raise CubeInputError(f"{path} に必要な配列がありません: {missing}")
    data, ra, dec, freq, meta = cube_mod.load_cube(path)
    if not isinstance(meta, dict):
        raise CubeInputError(f"{path} の meta が辞書ではありません")
    if meta.get("axes") != cube_mod.CUBE_AXES:
        raise CubeInputError(
            f"軸順序 {meta.get('axes')!r} には対応していません(対応: {cube_mod.CUBE_AXES!r})"
        )
    if meta.get("units") != cube_mod.CUBE_UNITS:
        raise CubeInputError(
            f"単位 {meta.get('units')!r} には対応していません(対応: {cube_mod.CUBE_UNITS!r})"
        )
    with open(path, "rb") as fh:
        file_sha = hashlib.sha256(fh.read()).hexdigest()
    sky = SkyCube(data, ra, dec, freq, identity={
        "kind": "file",
        "path": str(path),
        "file_sha256": file_sha,
        "file_meta": meta,
    })
    validate_sky_cube(sky)
    return sky


# ---------------------------------------------------------------------------
# 観測設定
# ---------------------------------------------------------------------------

@dataclass
class ObservingSetup:
    """固定点観測に必要な設定だけを集めたもの(天体キューブの作り方は含まない)"""

    seed: int
    pointing_ra_arcsec: float
    pointing_dec_arcsec: float
    spectrometer: SpectrometerConfig
    modulation: ModulationConfig
    observation: ObservationConfig
    noise: NoiseConfig

    @property
    def n_time(self) -> int:
        return int(round(self.observation.duration_s / self.modulation.dt_s))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, cfg: Config, pointing_ra_arcsec: float, pointing_dec_arcsec: float):
        return cls(
            seed=cfg.seed,
            pointing_ra_arcsec=float(pointing_ra_arcsec),
            pointing_dec_arcsec=float(pointing_dec_arcsec),
            spectrometer=cfg.spectrometer, modulation=cfg.modulation,
            observation=cfg.observation, noise=cfg.noise,
        )

    def validate(self) -> None:
        validate_observing(self.spectrometer, self.modulation, self.noise, self.n_time)


def setup_from_dict(data: dict) -> ObservingSetup:
    """ObservingSetup.to_dict() の辞書から復元する"""
    setup = ObservingSetup(
        seed=int(data["seed"]),
        pointing_ra_arcsec=float(data["pointing_ra_arcsec"]),
        pointing_dec_arcsec=float(data["pointing_dec_arcsec"]),
        spectrometer=_build(SpectrometerConfig, data["spectrometer"], "spectrometer"),
        modulation=_build(ModulationConfig, data["modulation"], "modulation"),
        observation=_build(ObservationConfig, data["observation"], "observation"),
        noise=_build(NoiseConfig, data["noise"], "noise"),
    )
    setup.validate()
    return setup


def load_observing_setup(path, overrides: Optional[dict] = None):
    """観測設定 YAML を読む。(ObservingSetup, 無視したセクション名のリスト) を返す

    使うセクションは seed / pointing / spectrometer / modulation / observation / noise。
    既存の config/default.yaml をそのまま渡せるよう source / cube / removal / evaluation は
    置いてあってもよいが、外部キューブの観測では使わない(戻り値で知らせる)。
    pointing は必須(null は「天体中心」の意味で、外部キューブでは定義できないためエラー)。
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} は辞書形式の YAML である必要があります")
    for key, value in (overrides or {}).items():
        _apply_override(data, key, value)
    unknown = set(data) - set(SETUP_SECTIONS) - set(IGNORED_SECTIONS)
    missing = set(SETUP_SECTIONS) - set(data)
    if unknown or missing:
        raise ValueError(f"観測設定のキーが不正です: 未知={sorted(unknown)}, 不足={sorted(missing)}")
    pointing = data["pointing"]
    if not isinstance(pointing, dict) or set(pointing) != {"ra_arcsec", "dec_arcsec"}:
        raise ValueError("pointing には ra_arcsec と dec_arcsec を指定してください")
    if pointing["ra_arcsec"] is None or pointing["dec_arcsec"] is None:
        raise ValueError(
            "外部キューブの観測では pointing.ra_arcsec / pointing.dec_arcsec を数値で指定してください"
            "(null は内部生成キューブの「天体中心」の意味で、外部キューブでは定義できません)"
        )
    setup = ObservingSetup(
        seed=int(data["seed"]),
        pointing_ra_arcsec=float(pointing["ra_arcsec"]),
        pointing_dec_arcsec=float(pointing["dec_arcsec"]),
        spectrometer=_build(SpectrometerConfig, data["spectrometer"], "spectrometer"),
        modulation=_build(ModulationConfig, data["modulation"], "modulation"),
        observation=_build(ObservationConfig, data["observation"], "observation"),
        noise=_build(NoiseConfig, data["noise"], "noise"),
    )
    setup.validate()
    return setup, [k for k in IGNORED_SECTIONS if k in data]


# ---------------------------------------------------------------------------
# キューブの軸と観測設定の対応
# ---------------------------------------------------------------------------

def match_frequency_grid(freq_ghz, spectrometer: SpectrometerConfig) -> cube_mod.FrequencyGrid:
    """キューブの周波数軸の上に分光器の観測窓(変調なし)を置く

    キューブの周波数軸は入力データとしてそのまま使い、作り直さない。
    次のどれかに当てはまればエラー:
      - キューブのチャンネル幅が分光器と違う
      - 変調なしの窓のチャンネルがキューブのチャンネルに重ならない(半端なずれ)
      - 変調なしの窓がキューブの周波数範囲に収まらない
    """
    freq = np.asarray(freq_ghz, dtype=float)
    n_ch = spectrometer.n_channels
    dnu = spectrometer.channel_width_mhz * 1e-3
    step = (freq[-1] - freq[0]) / (freq.size - 1)
    # 幅の違いが軸全体で 1e-6 ch 以上のずれになるなら不整合とする
    if abs(step - dnu) * (freq.size - 1) > AXIS_RTOL * dnu:
        raise CubeInputError(
            f"キューブのチャンネル幅 {step * 1e3:.9g} MHz が分光器の "
            f"{spectrometer.channel_width_mhz:.9g} MHz と一致しません(再標本化は行いません)"
        )
    f_first = spectrometer.center_freq_ghz - (n_ch - 1) / 2.0 * dnu   # 変調なしの窓の先頭チャンネル
    start_float = (f_first - freq[0]) / dnu
    start0 = int(round(start_float))
    if abs(start_float - start0) > AXIS_RTOL:
        raise CubeInputError(
            "分光器のチャンネルがキューブの周波数チャンネルに重なりません"
            f"(ずれ {start_float - start0:+.6f} ch)。spectrometer.center_freq_ghz を見直してください"
        )
    if start0 < 0 or start0 + n_ch > freq.size:
        raise CubeInputError(
            "キューブの周波数範囲が分光器の帯域に足りません: 必要 "
            f"{f_first:.6f} - {f_first + (n_ch - 1) * dnu:.6f} GHz、"
            f"キューブ {freq[0]:.6f} - {freq[-1]:.6f} GHz"
        )
    # make_frequency_grid と同じ式で理想格子を作り、軸全体が重なることを確かめる
    ideal = spectrometer.center_freq_ghz + (np.arange(freq.size) - start0 - (n_ch - 1) / 2.0) * dnu
    if np.max(np.abs(freq - ideal)) > AXIS_RTOL * dnu:
        raise CubeInputError("キューブの周波数軸が分光器のチャンネル格子と一致しません")
    return cube_mod.FrequencyGrid(freq, n_ch, start0, dnu, spectrometer.center_freq_ghz)


def check_modulation_band(grid: cube_mod.FrequencyGrid, shifts_ch) -> None:
    """全時刻の観測窓がキューブの周波数範囲に収まるか確認する。足りなければエラー"""
    lo = grid.start0 + int(np.min(shifts_ch))
    hi = grid.start0 + int(np.max(shifts_ch)) + grid.n_channels   # 終端(含まない)
    if lo < 0 or hi > grid.n_wide:
        f0 = grid.freq_ghz[0] + lo * grid.channel_width_ghz
        f1 = grid.freq_ghz[0] + (hi - 1) * grid.channel_width_ghz
        raise CubeInputError(
            "キューブの周波数範囲が変調後の観測窓に足りません: 必要 "
            f"{f0:.6f} - {f1:.6f} GHz(低周波側 {max(0, -lo)} ch・高周波側 {max(0, hi - grid.n_wide)} ch 不足)、"
            f"キューブ {grid.freq_ghz[0]:.6f} - {grid.freq_ghz[-1]:.6f} GHz。"
            "外挿・ゼロ埋めは行いません"
        )


def _check_in_axis(name, axis, value):
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1:
        # 1 画素だけの軸は画素の大きさが分からないので、座標が一致する場合だけ受け付ける
        if abs(value - axis[0]) > 1e-9:
            raise CubeInputError(
                f"{name} 軸が 1 画素だけなので、指向位置は {axis[0]} arcsec に一致させてください(指定 {value})"
            )
        return
    pix = (axis[-1] - axis[0]) / (axis.size - 1)
    lo, hi = axis[0] - pix / 2.0, axis[-1] + pix / 2.0
    slack = 1e-9 * pix
    if not (lo - slack <= value <= hi + slack):
        raise CubeInputError(
            f"指向位置 {name} = {value} arcsec がマップの範囲 [{lo}, {hi}] arcsec の外です"
            "(端の画素への丸めは行いません)"
        )


def check_pointing_in_map(sky: SkyCube, ra_arcsec: float, dec_arcsec: float) -> None:
    """指向位置がマップ内(端の画素の外側の縁まで)にあるか確認する"""
    if not (np.isfinite(ra_arcsec) and np.isfinite(dec_arcsec)):
        raise CubeInputError("指向位置が有限値ではありません")
    _check_in_axis("RA", sky.ra_arcsec, float(ra_arcsec))
    _check_in_axis("Dec", sky.dec_arcsec, float(dec_arcsec))


# ---------------------------------------------------------------------------
# 固定点観測
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """固定点観測の結果一式。truth_k / signal_tod / noise は正解情報(評価専用)"""

    setup: ObservingSetup
    grid: cube_mod.FrequencyGrid
    cube_identity: dict
    pixel: tuple                # スペクトルを取り出した画素 (i_ra, i_dec)
    pixel_ra_arcsec: float      # その画素の座標
    pixel_dec_arcsec: float
    truth_k: np.ndarray         # 正解スペクトル (n_wide,) [K]
    time_s: np.ndarray          # (n_time,) [s]
    shifts_ch: np.ndarray       # (n_time,) [ch]
    signal_tod: np.ndarray      # 天体信号のみ (n_time, n_channels) [K]
    noise: noise.NoiseComponents
    observed: np.ndarray        # 天体信号 + 全雑音 (n_time, n_channels) [K]


def observe(sky: SkyCube, setup: ObservingSetup) -> Observation:
    """キューブの 1 点を FMLO で観測した時系列を作る(入力は変更しない)

    乱数は config.spawn_rngs(seed) の成分別生成器を使う(既存の simulate と同じ)。
    """
    setup.validate()
    validate_sky_cube(sky)
    grid = match_frequency_grid(sky.freq_ghz, setup.spectrometer)
    check_pointing_in_map(sky, setup.pointing_ra_arcsec, setup.pointing_dec_arcsec)

    rngs = spawn_rngs(setup.seed)
    n_time, n_ch = setup.n_time, grid.n_channels
    dt = setup.modulation.dt_s

    # 1 点のスペクトル(最近傍画素)
    truth, pixel = cube_mod.extract_spectrum(
        sky.data, sky.ra_arcsec, sky.dec_arcsec, setup.pointing_ra_arcsec, setup.pointing_dec_arcsec
    )

    # FMLO 変調
    shifts = modulation.shifts_from_config(setup.modulation, n_time, rngs["modulation"])
    check_modulation_band(grid, shifts)
    signal_tod = modulation.modulate(truth, shifts, grid.start0, n_ch)

    # 雑音(成分ごとに生成して登録する。スキャンと共用の関数)
    comps = noise.make_noise_components(setup.noise, setup.spectrometer, grid, shifts, dt, rngs)

    return Observation(
        setup=setup, grid=grid, cube_identity=dict(sky.identity), pixel=pixel,
        pixel_ra_arcsec=float(sky.ra_arcsec[pixel[0]]), pixel_dec_arcsec=float(sky.dec_arcsec[pixel[1]]),
        truth_k=truth, time_s=np.arange(n_time) * dt, shifts_ch=shifts,
        signal_tod=signal_tod, noise=comps, observed=signal_tod + comps.total(),
    )
