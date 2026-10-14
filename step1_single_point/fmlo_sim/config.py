"""設定ファイル(YAML)の読み込み

既定値は config/default.yaml だけに置く。ここの dataclass は型とキーの定義のみで、
YAML にキーの過不足があればエラーにする。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, get_type_hints

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

# 受信機の観測帯域 [GHz]。輝線の中心周波数はこの範囲に制限する
BAND_MIN_GHZ = 125.0
BAND_MAX_GHZ = 163.0

MEAN_MODES = ("none", "channel_mean")
PATTERNS = ("zigzag", "random")


ATMOSPHERE_MODELS = ("laminar", "colored", "turbulent")


@dataclass
class SourceConfig:
    sigma_arcsec: float
    line_fwhm_kms: float
    center_freq_ghz: float
    peak_k: float
    center_ra_arcsec: float
    center_dec_arcsec: float


@dataclass
class CubeConfig:
    half_extent_arcsec: float
    pixel_arcsec: float
    margin_ch: int


@dataclass
class PointingConfig:
    ra_arcsec: Optional[float]
    dec_arcsec: Optional[float]


@dataclass
class SpectrometerConfig:
    n_channels: int
    channel_width_mhz: float
    center_freq_ghz: float


@dataclass
class ModulationConfig:
    pattern: str
    dt_s: float
    half_width_ch: int
    zigzag_step_ch: int


@dataclass
class ObservationConfig:
    duration_s: float


@dataclass
class ThermalConfig:
    enabled: bool
    tsys_k: float


@dataclass
class AtmosphereConfig:
    enabled: bool
    alpha: float
    rms_k: float
    offset_k: float
    slope_frac: float
    model: str = "laminar"


@dataclass
class NoiseConfig:
    thermal: ThermalConfig
    atmosphere: AtmosphereConfig


@dataclass
class RemovalConfig:
    n_components: int
    mean_mode: str


@dataclass
class EvaluationConfig:
    line_free_n_fwhm: float


@dataclass
class VisualizationConfig:
    cube3d_half_range_n_fwhm: float
    cube3d_iso_fractions: List[float]
    cube3d_opacity: float
    cube3d_max_freq_points: int


@dataclass
class Config:
    seed: int
    source: SourceConfig
    cube: CubeConfig
    pointing: PointingConfig
    spectrometer: SpectrometerConfig
    modulation: ModulationConfig
    observation: ObservationConfig
    noise: NoiseConfig
    removal: RemovalConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig

    @property
    def n_time(self) -> int:
        """時間サンプル数"""
        return int(round(self.observation.duration_s / self.modulation.dt_s))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, data, path=""):
    """辞書から dataclass を組み立てる。キーの過不足はエラー"""
    if not isinstance(data, dict):
        raise ValueError(f"設定 '{path}' は辞書である必要があります")
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)
    names = [f.name for f in fields]
    required = [f.name for f in fields if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    unknown = set(data) - set(names)
    missing = set(required) - set(data)
    if unknown or missing:
        raise ValueError(
            f"設定 '{path or 'root'}' のキーが不正です: 未知={sorted(unknown)}, 不足={sorted(missing)}"
        )
    kwargs = {}
    for f in fields:
        name = f.name
        typ = hints[name]
        if name in data:
            value = data[name]
            if dataclasses.is_dataclass(typ):
                value = _build(typ, value, f"{path}.{name}".lstrip("."))
            elif typ is float and value is not None:
                value = float(value)
            kwargs[name] = value
        elif f.default is not dataclasses.MISSING:
            kwargs[name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            kwargs[name] = f.default_factory()
    return cls(**kwargs)


def _apply_override(data: dict, key: str, value) -> None:
    """'noise.atmosphere.offset_k' のようなドット区切りのキーで値を上書きする"""
    parts = key.split(".")
    node = data
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            raise KeyError(f"設定キー '{key}' は存在しません")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"設定キー '{key}' は存在しません")
    node[parts[-1]] = value


def parse_set_args(items) -> dict:
    """コマンドラインの 'key=value' のリストを上書き用の辞書にする"""
    overrides = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set は key=value の形で指定してください: {item}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def validate(cfg: Config) -> None:
    if not BAND_MIN_GHZ <= cfg.source.center_freq_ghz <= BAND_MAX_GHZ:
        raise ValueError(
            f"source.center_freq_ghz は {BAND_MIN_GHZ}–{BAND_MAX_GHZ} GHz の範囲で指定してください"
        )
    if cfg.modulation.pattern not in PATTERNS:
        raise ValueError(f"modulation.pattern は {PATTERNS} のいずれかです")
    if cfg.removal.mean_mode not in MEAN_MODES:
        raise ValueError(f"removal.mean_mode は {MEAN_MODES} のいずれかです")
    if cfg.noise.atmosphere.model not in ATMOSPHERE_MODELS:
        raise ValueError(f"noise.atmosphere.model は {ATMOSPHERE_MODELS} のいずれかです")
    if cfg.modulation.half_width_ch < 0 or cfg.cube.margin_ch < 0:
        raise ValueError("half_width_ch と margin_ch は 0 以上である必要があります")
    if cfg.modulation.zigzag_step_ch < 1:
        raise ValueError("modulation.zigzag_step_ch は 1 以上である必要があります")
    if 2 * cfg.modulation.half_width_ch >= cfg.spectrometer.n_channels:
        raise ValueError("変調幅が分光計の帯域以上になっています")
    viz = cfg.visualization
    if viz.cube3d_half_range_n_fwhm <= 0:
        raise ValueError("visualization.cube3d_half_range_n_fwhm は正の値である必要があります")
    if not isinstance(viz.cube3d_iso_fractions, list) or not viz.cube3d_iso_fractions:
        raise ValueError("visualization.cube3d_iso_fractions は1つ以上の値のリストで指定してください")
    if not all(isinstance(x, (int, float)) and 0 < x <= 1 for x in viz.cube3d_iso_fractions):
        raise ValueError("visualization.cube3d_iso_fractions の値は 0 より大きく 1 以下である必要があります")
    if not 0 <= viz.cube3d_opacity <= 1:
        raise ValueError("visualization.cube3d_opacity は 0〜1 の範囲で指定してください")
    if viz.cube3d_max_freq_points < 2:
        raise ValueError("visualization.cube3d_max_freq_points は 2 以上である必要があります")
    if cfg.n_time < 2:
        raise ValueError("観測時間が短すぎます")


def load_config(path=None, overrides: Optional[dict] = None) -> Config:
    """YAML を読み込んで Config を返す

    overrides: {'source.line_fwhm_kms': 100.0} のようなドット区切りキーの上書き
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        _apply_override(data, key, value)
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


# 乱数を使う成分の名前。順番を変えると既存成分の乱数列が変わるので、追加は末尾に行う
RNG_NAMES = ("modulation", "thermal", "atmosphere")


def spawn_rngs(seed: int) -> dict:
    """シードから成分ごとに独立した乱数生成器を作る

    ある成分の設定を変えても、他の成分の乱数列は変わらない。
    """
    children = np.random.SeedSequence(seed).spawn(len(RNG_NAMES))
    return {name: np.random.default_rng(s) for name, s in zip(RNG_NAMES, children)}
