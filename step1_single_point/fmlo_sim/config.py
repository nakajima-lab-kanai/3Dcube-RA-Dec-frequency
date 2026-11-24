"""設定ファイル(YAML)の読み込み

既定値は config/default.yaml だけに置く。ここの dataclass は型とキーの定義のみで、
YAML にキーの過不足があればエラーにする。
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, get_type_hints

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

# 受信機の観測帯域 [GHz]。輝線の中心周波数はこの範囲に制限する
BAND_MIN_GHZ = 125.0
BAND_MAX_GHZ = 163.0

MEAN_MODES = ("none", "channel_mean")
PATTERNS = ("zigzag", "random")


# laminar: 穏やかなドリフト / colored(= turbulent): f^-alpha の有色雑音 /
# random_walk: 正規乱数の累積和 / external: 外部シミュレータ(GATO など)の配列を読み込む
ATMOSPHERE_MODELS = ("laminar", "colored", "turbulent", "random_walk", "external")
REMOVAL_METHODS = ("simple_pca", "iterative_pca")
SCAN_MODES = ("single_point", "x_scan", "y_scan", "basket_weave")
MAPPING_KERNELS = ("gaussian",)
BASKET_WEAVE_METHODS = ("weighted_mean", "fourier")
OFF_INTERPOLATIONS = ("linear", "nearest")


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
    # zigzag の 1 周期ごとに [-floor(step/2), +floor(step/2)] の整数オフセットを足し、
    # 変調が厳密な周期にならないようにする(既定 false = 従来と同じ)
    cycle_jitter: bool = False

    @property
    def jitter_max_ch(self) -> int:
        """cycle_jitter で足されるオフセットの最大値 [ch]"""
        return int(self.zigzag_step_ch) // 2 if self.cycle_jitter else 0

    @property
    def max_abs_shift_ch(self) -> int:
        """シフト量の絶対値の上限 [ch](周波数グリッドの幅を決める)"""
        return int(self.half_width_ch) + self.jitter_max_ch


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
    external_path: Optional[str] = None   # model = external のときの配列ファイル(.npy / .npz)


@dataclass
class InstrumentConfig:
    """装置由来の 1/f^alpha 雑音。チャンネル(IF)に固定され、変調で動かない"""

    enabled: bool = False
    alpha: float = 1.0
    rms_k: float = 0.1


@dataclass
class NoiseConfig:
    thermal: ThermalConfig
    atmosphere: AtmosphereConfig
    instrument: InstrumentConfig = dataclasses.field(default_factory=InstrumentConfig)


@dataclass
class RemovalConfig:
    n_components: int
    mean_mode: str
    method: str = "simple_pca"   # simple_pca(1 回)/ iterative_pca(反復)
    max_iter: int = 20           # iterative_pca の最大反復回数
    tol: float = 0.005           # |A_i - A_(i-1)|_F / |A_i|_F がこれ未満で収束
    cutoff_sigma: float = 5.0    # 輝線モデルに採用する S/N の閾値(0 なら全チャンネルを採用)
    model_smooth_ch: int = 1     # スキャンの輝線モデルで、判定前に周波数方向に平滑化する幅 [ch]


@dataclass
class EvaluationConfig:
    line_free_n_fwhm: float


@dataclass
class TelescopeConfig:
    """望遠鏡ビーム。beam_convolve = true なら正解キューブを空間方向にガウシアンで畳み込む"""

    beam_convolve: bool = False
    beam_fwhm_arcsec: float = 12.0   # 野辺山 45m・150 GHz を想定 要確認


@dataclass
class ScanConfig:
    """1 ビーム OTF スキャンの軌跡(段階 2-B)。run_step1.py では使わない"""

    mode: str = "single_point"         # single_point / x_scan / y_scan / basket_weave
    map_size_arcsec: float = 400.0     # スキャンする正方形領域の一辺 [arcsec]
    scan_speed_arcsec_s: float = 20.0  # スキャン速度 [arcsec/s]
    row_spacing_arcsec: float = 10.0   # スキャンライン(行)の間隔 [arcsec]
    center_ra_arcsec: float = 0.0      # スキャン領域の中心 [arcsec]
    center_dec_arcsec: float = 0.0


@dataclass
class MappingConfig:
    """タイムストリームから出力キューブを作るグリッディングの設定(段階 2-B)"""

    grid_pixel_arcsec: float = 10.0
    kernel: str = "gaussian"
    kernel_fwhm_arcsec: float = 15.0
    kernel_support_fwhm: float = 1.5             # カーネルを打ち切る半径(kernel_fwhm の何倍か)
    basket_weave_method: str = "weighted_mean"   # weighted_mean / fourier


@dataclass
class OtfConfig:
    """従来の OTF(変調なし・OFF 点を引く)の比較観測の設定。run_step2_otf_vs_fmlo.py だけが使う"""

    off_interval_rows: int = 1         # OFF 点を何行ごとに観測するか 要確認
    off_integration_s: float = 5.0     # 1 回の OFF 点の積分時間 [s] 要確認
    off_interpolation: str = "linear"  # linear(前後の OFF を時間で線形補間)/ nearest
    baseline_order: int = 1            # 輝線のないチャンネルで引く多項式ベースラインの次数(-1 なら引かない)


@dataclass
class ComparisonConfig:
    """OTF と FMLO の比較図の積分範囲(輝線中心から ± 何 FWHM)。run_step2_otf_vs_fmlo.py だけが使う"""

    single_half_width_fwhm: float = 1.0   # 単一スキャン(論文 Fig. 14 は 50.5 km/s 幅)
    basket_half_width_fwhm: float = 0.6   # basket weave(論文は 30.5 km/s 幅 = 約 0.6 倍)


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
    telescope: TelescopeConfig = dataclasses.field(default_factory=TelescopeConfig)
    scan: ScanConfig = dataclasses.field(default_factory=ScanConfig)
    mapping: MappingConfig = dataclasses.field(default_factory=MappingConfig)
    otf: OtfConfig = dataclasses.field(default_factory=OtfConfig)
    comparison: ComparisonConfig = dataclasses.field(default_factory=ComparisonConfig)

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


def _apply_override(data: dict, key: str, value, schema=None) -> None:
    """'noise.atmosphere.offset_k' のようなドット区切りのキーで値を上書きする

    schema(dataclass)を渡すと、YAML に書かれていなくても dataclass に定義された
    キー(既定値を持つ新しい設定項目)は上書きできる。どちらにも無いキーはエラー。
    """
    parts = key.split(".")
    node = data
    cls = schema
    for i, p in enumerate(parts):
        field_cls = None
        if cls is not None:
            hints = get_type_hints(cls)
            if p not in hints:
                raise KeyError(f"設定キー '{key}' は存在しません")
            field_cls = hints[p] if dataclasses.is_dataclass(hints[p]) else None
        last = i == len(parts) - 1
        if last:
            if p not in node and cls is None:
                raise KeyError(f"設定キー '{key}' は存在しません")
            node[p] = value
            return
        if not isinstance(node.get(p), dict):
            if field_cls is None or p in node:
                raise KeyError(f"設定キー '{key}' は存在しません")
            node[p] = {}   # YAML に無いセクション(既定値を持つもの)を作る
        node = node[p]
        cls = field_cls


def parse_set_args(items) -> dict:
    """コマンドラインの 'key=value' のリストを上書き用の辞書にする"""
    overrides = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set は key=value の形で指定してください: {item}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def validate_observing(spectrometer: SpectrometerConfig, modulation: ModulationConfig,
                       noise: NoiseConfig, n_time: int) -> None:
    """観測処理(分光器・変調・雑音・観測時間)だけに関わる検証

    天体キューブの作り方に依存しないので、内部生成キューブと外部入力キューブで共用する。
    """
    if modulation.pattern not in PATTERNS:
        raise ValueError(f"modulation.pattern は {PATTERNS} のいずれかです")
    if noise.atmosphere.model not in ATMOSPHERE_MODELS:
        raise ValueError(f"noise.atmosphere.model は {ATMOSPHERE_MODELS} のいずれかです")
    if modulation.half_width_ch < 0:
        raise ValueError("half_width_ch と margin_ch は 0 以上である必要があります")
    if modulation.zigzag_step_ch < 1:
        raise ValueError("modulation.zigzag_step_ch は 1 以上である必要があります")
    if modulation.cycle_jitter and modulation.pattern != "zigzag":
        raise ValueError("modulation.cycle_jitter は pattern = zigzag のときだけ使えます")
    if 2 * modulation.max_abs_shift_ch >= spectrometer.n_channels:
        raise ValueError("変調幅が分光計の帯域以上になっています")
    atm = noise.atmosphere
    if atm.enabled and atm.model == "external" and not atm.external_path:
        raise ValueError("noise.atmosphere.model = external には noise.atmosphere.external_path が必要です")
    if noise.instrument.rms_k < 0:
        raise ValueError("noise.instrument.rms_k は 0 以上である必要があります")
    if n_time < 2:
        raise ValueError("観測時間が短すぎます")


def validate(cfg: Config) -> None:
    if not BAND_MIN_GHZ <= cfg.source.center_freq_ghz <= BAND_MAX_GHZ:
        raise ValueError(
            f"source.center_freq_ghz は {BAND_MIN_GHZ}–{BAND_MAX_GHZ} GHz の範囲で指定してください"
        )
    if cfg.removal.mean_mode not in MEAN_MODES:
        raise ValueError(f"removal.mean_mode は {MEAN_MODES} のいずれかです")
    if cfg.cube.margin_ch < 0:
        raise ValueError("half_width_ch と margin_ch は 0 以上である必要があります")
    r = cfg.removal
    if r.method not in REMOVAL_METHODS:
        raise ValueError(f"removal.method は {REMOVAL_METHODS} のいずれかです")
    if r.n_components < 0:
        raise ValueError("removal.n_components は 0 以上である必要があります")
    if r.max_iter < 1 or not r.tol > 0 or r.cutoff_sigma < 0 or r.model_smooth_ch < 1:
        raise ValueError("removal.max_iter >= 1、tol > 0、cutoff_sigma >= 0、model_smooth_ch >= 1 が必要です")
    if cfg.telescope.beam_convolve and not cfg.telescope.beam_fwhm_arcsec > 0:
        raise ValueError("telescope.beam_fwhm_arcsec は正の値である必要があります")
    sc = cfg.scan
    if sc.mode not in SCAN_MODES:
        raise ValueError(f"scan.mode は {SCAN_MODES} のいずれかです")
    if not (sc.map_size_arcsec > 0 and sc.scan_speed_arcsec_s > 0 and sc.row_spacing_arcsec > 0):
        raise ValueError("scan.map_size_arcsec / scan_speed_arcsec_s / row_spacing_arcsec は正の値です")
    m = cfg.mapping
    if m.kernel not in MAPPING_KERNELS:
        raise ValueError(f"mapping.kernel は {MAPPING_KERNELS} のいずれかです")
    if m.basket_weave_method not in BASKET_WEAVE_METHODS:
        raise ValueError(f"mapping.basket_weave_method は {BASKET_WEAVE_METHODS} のいずれかです")
    if not (m.grid_pixel_arcsec > 0 and m.kernel_fwhm_arcsec > 0 and m.kernel_support_fwhm > 0):
        raise ValueError("mapping.grid_pixel_arcsec / kernel_fwhm_arcsec / kernel_support_fwhm は正の値です")
    o = cfg.otf
    if o.off_interval_rows < 1 or not o.off_integration_s > 0 or o.baseline_order < -1:
        raise ValueError("otf.off_interval_rows >= 1、off_integration_s > 0、baseline_order >= -1 が必要です")
    if not (cfg.comparison.single_half_width_fwhm > 0 and cfg.comparison.basket_half_width_fwhm > 0):
        raise ValueError("comparison の積分範囲は正の値です")
    if o.off_interpolation not in OFF_INTERPOLATIONS:
        raise ValueError(f"otf.off_interpolation は {OFF_INTERPOLATIONS} のいずれかです")
    validate_observing(cfg.spectrometer, cfg.modulation, cfg.noise, cfg.n_time)


def config_from_dict(data: dict, overrides: Optional[dict] = None) -> Config:
    """YAML を読んだ辞書(と上書き)から Config を作って検証する。data は変更しない"""
    data = copy.deepcopy(data)
    for key, value in (overrides or {}).items():
        _apply_override(data, key, value, schema=Config)
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


def read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} は辞書形式の YAML である必要があります")
    return data


def load_config(path=None, overrides: Optional[dict] = None) -> Config:
    """YAML を読み込んで Config を返す

    overrides: {'source.line_fwhm_kms': 100.0} のようなドット区切りキーの上書き
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return config_from_dict(read_yaml(path), overrides)


# 乱数を使う成分の名前。順番を変えると既存成分の乱数列が変わるので、追加は末尾に行う
# (SeedSequence.spawn の i 番目の子は、全体の個数によらず同じになる)
RNG_NAMES = ("modulation", "thermal", "atmosphere", "instrument")

# スキャンの各パス(x_scan / y_scan)の乱数を、固定点観測(spawn_rngs(seed))と
# 重ならない系列から作るための spawn_key のオフセット
_PASS_KEY_OFFSET = 1000


def spawn_rngs(seed: int, pass_index: Optional[int] = None) -> dict:
    """シードから成分ごとに独立した乱数生成器を作る

    ある成分の設定を変えても、他の成分の乱数列は変わらない。
    pass_index を与えると、スキャンのパスごとに別の(互いに独立な)乱数列になる。
    """
    if pass_index is None:
        root = np.random.SeedSequence(seed)
    else:
        root = np.random.SeedSequence(seed, spawn_key=(_PASS_KEY_OFFSET + int(pass_index),))
    children = root.spawn(len(RNG_NAMES))
    return {name: np.random.default_rng(s) for name, s in zip(RNG_NAMES, children)}
