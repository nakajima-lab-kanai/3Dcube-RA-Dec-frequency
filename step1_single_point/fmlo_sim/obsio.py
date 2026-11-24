"""模擬観測データ(固定点・時間 × 分光器チャンネル)の保存と再読込

1 ファイル(.npz)に、観測時系列・正解情報・軸・設定・来歴をまとめて保存する。

保存精度:
  浮動小数の配列はすべて float64 のまま保存する(丸めなし)。したがって再読込した配列は
  保存前とビット単位で一致し、再読込後の解析結果も直接解析した結果と完全に一致する。
  (既存の run_step1.py の timestream.npz は float32 に丸めるので、こちらとは別の形式)

正解情報の分離:
  load_observation() は観測データ(ObservedData)と正解情報(ObservationTruth)を別々の
  オブジェクトで返す。推定計算には ObservedData だけを渡し、正解は評価でだけ使うこと。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import cube as cube_mod
from .observation import Observation, SkyCube

FORMAT_NAME = "fmlo_sim.fixed_point_observation"
FORMAT_VERSION = 1
FLOAT_DTYPE = "float64"

_TRUTH_KEYS = ("truth_k", "signal_tod")
_NOISE_PREFIX = "noise_"


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refuse_overwrite(paths, overwrite: bool) -> None:
    """既存ファイルがあれば、overwrite=True でない限りエラーにする"""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"出力先が既に存在します: {existing}。別の出力先を指定するか、--overwrite を付けてください"
        )


def _f64(a):
    a = np.asarray(a)
    if a.dtype != np.float64:
        raise TypeError(f"float64 の配列を期待しましたが {a.dtype} です")
    return a


def save_observation(path, obs: Observation, sky: SkyCube, provenance: Optional[dict] = None,
                     overwrite: bool = False) -> dict:
    """観測データを保存し、書き込んだ meta(辞書)を返す

    provenance : コードのコミット・変更状態・実行環境など(呼び出し側で集めた辞書)。
                 None なら「未取得」と理由を記録する。
    """
    path = Path(path)
    refuse_overwrite([path], overwrite)
    grid = obs.grid
    n_time, n_ch = obs.observed.shape
    noise_names = list(obs.noise.components)
    window_start = (grid.start0 + np.asarray(obs.shifts_ch)).astype(np.int64)

    meta = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "precision": {
            "float_arrays": FLOAT_DTYPE,
            "note": "浮動小数の配列は丸めずに float64 で保存。再読込後は保存前とビット単位で一致する",
        },
        "units": {
            "observed": "K", "signal_tod": "K", "noise_*": "K", "truth_k": "K",
            "time_s": "s", "shifts_ch": "channel", "freq_wide_ghz": "GHz",
            "cube_ra_arcsec": "arcsec", "cube_dec_arcsec": "arcsec",
        },
        "axes": {
            "observed": "[i_time, i_channel](時間 × 分光器チャンネル)",
            "signal_tod": "[i_time, i_channel]",
            "noise_*": "[i_time, i_channel]",
            "truth_k": "[i_freq](キューブの周波数軸 freq_wide_ghz 上)",
            "frequency_mapping": "時刻 t・チャンネル c の空の周波数 = "
                                 "freq_wide_ghz[window_start_index[t] + c]、"
                                 "window_start_index[t] = start0 + shifts_ch[t](正 = 高周波側)",
        },
        "cube": {
            "identity": obs.cube_identity,
            "content_sha256": sky.content_sha256(),
            "shape": list(sky.data.shape),
            "dtype": str(sky.data.dtype),
            "axes": cube_mod.CUBE_AXES,
            "units": dict(cube_mod.CUBE_UNITS),
        },
        "pointing": {
            "requested_ra_arcsec": obs.setup.pointing_ra_arcsec,
            "requested_dec_arcsec": obs.setup.pointing_dec_arcsec,
            "sampling": "nearest_pixel",
            "pixel_index": [int(obs.pixel[0]), int(obs.pixel[1])],
            "pixel_ra_arcsec": obs.pixel_ra_arcsec,
            "pixel_dec_arcsec": obs.pixel_dec_arcsec,
        },
        "grid": {
            "n_channels": int(grid.n_channels),
            "n_wide": int(grid.n_wide),
            "start0": int(grid.start0),
            "channel_width_ghz": float(grid.channel_width_ghz),
            "center_freq_ghz": float(grid.center_freq_ghz),
        },
        "n_time": int(n_time),
        "seed": int(obs.setup.seed),
        "setup": obs.setup.to_dict(),
        "noise_components": noise_names,
        "composition": "observed = signal_tod + (0 + noise_<1> + noise_<2> + ...)"
                       "(noise_components の順に加算)",
        "truth_arrays": [*_TRUTH_KEYS, *[_NOISE_PREFIX + n for n in noise_names]],
        "provenance": provenance if provenance is not None else {
            "status": "未取得", "reason": "save_observation に provenance が渡されなかった",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        observed=_f64(obs.observed),
        time_s=_f64(obs.time_s),
        shifts_ch=np.asarray(obs.shifts_ch).astype(np.int64),
        window_start_index=window_start,
        freq_wide_ghz=_f64(grid.freq_ghz),
        cube_ra_arcsec=np.asarray(sky.ra_arcsec),
        cube_dec_arcsec=np.asarray(sky.dec_arcsec),
        truth_k=_f64(obs.truth_k),
        signal_tod=_f64(np.asarray(obs.signal_tod, dtype=np.float64)),
        **{_NOISE_PREFIX + n: _f64(obs.noise.components[n]) for n in noise_names},
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return meta


@dataclass
class ObservedData:
    """推定計算に渡してよい情報(観測時系列・シフト量・軸)。正解は含まない"""

    observed: np.ndarray        # (n_time, n_channels) [K]
    time_s: np.ndarray
    shifts_ch: np.ndarray
    freq_wide_ghz: np.ndarray
    grid: cube_mod.FrequencyGrid
    meta: dict
    path: str
    file_sha256: str

    @property
    def n_time(self) -> int:
        return self.observed.shape[0]


@dataclass
class ObservationTruth:
    """評価でだけ使う正解情報"""

    truth_k: np.ndarray         # 正解スペクトル (n_wide,) [K]
    signal_tod: np.ndarray      # 天体信号のみ (n_time, n_channels) [K]
    noise: dict                 # 成分名 → (n_time, n_channels) [K](保存時の順序)


def load_observation(path):
    """save_observation の形式を読む。(ObservedData, ObservationTruth) を返す"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        if "meta" not in f.files:
            raise ValueError(f"{path} は観測データの形式ではありません(meta がありません)")
        meta = json.loads(str(f["meta"]))
        if meta.get("format") != FORMAT_NAME or meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path} の形式 {meta.get('format')!r} v{meta.get('format_version')!r} には対応していません"
            )
        arrays = {k: f[k] for k in f.files if k != "meta"}

    g = meta["grid"]
    observed = arrays["observed"]
    shifts = arrays["shifts_ch"]
    freq = arrays["freq_wide_ghz"]
    n_time, n_ch = observed.shape
    if (n_ch != g["n_channels"] or freq.size != g["n_wide"] or shifts.shape != (n_time,)
            or arrays["time_s"].shape != (n_time,) or n_time != meta["n_time"]):
        raise ValueError(f"{path} の配列の形が meta と整合しません")
    if not np.array_equal(arrays["window_start_index"], g["start0"] + shifts):
        raise ValueError(f"{path} の window_start_index が start0 + shifts_ch と一致しません")

    grid = cube_mod.FrequencyGrid(freq, g["n_channels"], g["start0"],
                                  g["channel_width_ghz"], g["center_freq_ghz"])
    data = ObservedData(
        observed=observed, time_s=arrays["time_s"], shifts_ch=shifts, freq_wide_ghz=freq,
        grid=grid, meta=meta, path=str(path), file_sha256=sha256_of_file(path),
    )
    truth = ObservationTruth(
        truth_k=arrays["truth_k"], signal_tod=arrays["signal_tod"],
        noise={n: arrays[_NOISE_PREFIX + n] for n in meta["noise_components"]},
    )
    for name, tod in {"signal_tod": truth.signal_tod, **truth.noise}.items():
        if tod.shape != observed.shape:
            raise ValueError(f"{path} の {name} の形が observed と一致しません")
    if truth.truth_k.shape != freq.shape:
        raise ValueError(f"{path} の truth_k の形が周波数軸と一致しません")
    return data, truth
