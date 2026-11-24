"""保存した観測データの解析(正解を使わない)と評価(正解を使う)を分けて行う

analyze()  : 観測時系列・シフト量・格子情報だけを使う。正解は引数に取らない。
evaluate() : analyze() の結果を正解スペクトルと比べる。正解を参照するのはここだけ。

解析の中身は既存の処理をそのまま呼ぶ:
  基本解析 = pipeline.reduce_observed_all_modes(run_step1.py と同じ 3 通り)
  2 回推定 = iterative.run_two_step_estimation(run_fig6.py と同じ)
論文 Fig. 6 再現(taniguchi2020.py)の解析仕様とは別物で、ここでは使わない。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from . import evaluation, iterative, pipeline
from .config import MEAN_MODES, EvaluationConfig, RemovalConfig, _apply_override, _build
from .obsio import ObservationTruth, ObservedData


@dataclass
class AnalysisConfig:
    removal: RemovalConfig
    evaluation: EvaluationConfig


def load_analysis_config(path, overrides: Optional[dict] = None) -> AnalysisConfig:
    """設定 YAML の removal / evaluation セクションだけを読む(他のセクションは見ない)"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if key.split(".")[0] not in ("removal", "evaluation"):
            raise KeyError(f"解析で上書きできるのは removal.* と evaluation.* だけです: {key}")
        _apply_override(data, key, value)
    missing = {"removal", "evaluation"} - set(data)
    if missing:
        raise ValueError(f"解析設定に必要なセクションがありません: {sorted(missing)}")
    cfg = AnalysisConfig(_build(RemovalConfig, data["removal"], "removal"),
                         _build(EvaluationConfig, data["evaluation"], "evaluation"))
    if cfg.removal.mean_mode not in MEAN_MODES:
        raise ValueError(f"removal.mean_mode は {MEAN_MODES} のいずれかです")
    return cfg


@dataclass
class AnalysisResult:
    n_components: int
    mean_mode: str                              # 2 回推定で使う平均の扱い(基本解析は全モード実行)
    basic: dict                                 # label → pipeline.BlindReduction
    two_step: Optional[iterative.TwoStepResult]
    cutoff_sigma: Optional[float]


def analyze(data: ObservedData, n_components: int, mean_mode: str,
            two_step: bool = False, cutoff_sigma: Optional[float] = None) -> AnalysisResult:
    """観測データだけから復元スペクトルを推定する(入力は変更しない)"""
    grid = data.grid
    basic = pipeline.reduce_observed_all_modes(
        data.observed, data.shifts_ch, grid.start0, grid.n_wide, n_components
    )
    two = None
    if two_step:
        if cutoff_sigma is None:
            raise ValueError("2 回推定には cutoff_sigma が必要です")
        two = iterative.run_two_step_estimation(
            observed_tod=data.observed, shifts_ch=data.shifts_ch, start0=grid.start0,
            n_wide=grid.n_wide, n_components=n_components, mean_mode=mean_mode,
            cutoff_sigma=cutoff_sigma,
        )
    return AnalysisResult(n_components, mean_mode, basic, two, cutoff_sigma if two_step else None)


# ---------------------------------------------------------------------------
# 評価(正解を参照する)
# ---------------------------------------------------------------------------

@dataclass
class LineDefinition:
    """評価で「輝線領域」「輝線のない領域」を分ける定義

    kind = 'gaussian' : 中心 ±(n_fwhm × FWHM)。既存の evaluation.evaluate と同じ
    kind = 'windows'  : 周波数窓 [GHz] を明示。積分強度の速度換算に ref_freq_ghz を使う
    """

    kind: str
    center_freq_ghz: Optional[float] = None
    line_fwhm_kms: Optional[float] = None
    n_fwhm: Optional[float] = None
    windows_ghz: Optional[list] = None
    ref_freq_ghz: Optional[float] = None
    origin: str = ""        # どこから来た定義か(記録用)

    def to_dict(self) -> dict:
        if self.kind == "gaussian":
            return {"kind": "gaussian", "center_freq_ghz": self.center_freq_ghz,
                    "line_fwhm_kms": self.line_fwhm_kms, "n_fwhm": self.n_fwhm, "origin": self.origin}
        return {"kind": "windows", "windows_ghz": [list(w) for w in self.windows_ghz],
                "ref_freq_ghz": self.ref_freq_ghz, "origin": self.origin}


def line_definition_from_source(meta: dict, n_fwhm: float) -> LineDefinition:
    """観測データに記録されたキューブの生成条件(ガウシアン天体)から輝線領域を決める

    生成条件が記録されていないキューブ(外部で作ったものなど)ではエラー。推測はしない。
    """
    identity = meta["cube"]["identity"]
    source = None
    if identity.get("kind") == "internal_gaussian":
        source = identity.get("source")
    elif identity.get("kind") == "file":
        source = ((identity.get("file_meta") or {}).get("config") or {}).get("source")
    if not source:
        raise ValueError(
            "このキューブにはガウシアン天体の生成条件が記録されていません。"
            "輝線領域は --line-window-ghz で明示してください"
        )
    return LineDefinition("gaussian", center_freq_ghz=float(source["center_freq_ghz"]),
                          line_fwhm_kms=float(source["line_fwhm_kms"]), n_fwhm=float(n_fwhm),
                          origin="キューブに記録された生成条件 (source)")


def _evaluate_spectrum(restored_k, hits, data: ObservedData, truth: ObservationTruth,
                       line_def: Optional[LineDefinition]) -> dict:
    grid = data.grid
    out = dict(evaluation.residual_summary(restored_k, truth.truth_k, hits, data.n_time))
    if line_def is None:
        out["line_metrics"] = None
        out["line_metrics_reason"] = "輝線領域の定義が指定されなかったため未計算"
        return out
    if line_def.kind == "gaussian":
        line = evaluation.evaluate(
            restored_k, truth.truth_k, hits, data.n_time, grid.freq_ghz,
            line_def.center_freq_ghz, line_def.line_fwhm_kms, line_def.n_fwhm, grid.channel_width_ghz,
        )
    elif line_def.kind == "windows":
        in_line, line_free = evaluation.window_masks(grid.freq_ghz, line_def.windows_ghz)
        line = evaluation.evaluate_masked(
            restored_k, truth.truth_k, hits, data.n_time, in_line, line_free,
            grid.channel_width_ghz, line_def.ref_freq_ghz,
        )
    else:
        raise ValueError(f"未知の輝線定義です: {line_def.kind}")
    out["line_metrics"] = line
    out["line_metrics_reason"] = None
    return out


def evaluate(result: AnalysisResult, data: ObservedData, truth: ObservationTruth,
             line_def: Optional[LineDefinition] = None) -> dict:
    """解析結果を正解と比べる。評価値の辞書(JSON にできる)を返す"""
    summary = {
        "basic": {
            label: {
                "n_components": r.n_components, "mean_mode": r.mean_mode,
                "explained_ratio": [float(x) for x in r.pca.explained_ratio],
                **_evaluate_spectrum(r.restored_k, r.hits, data, truth, line_def),
            }
            for label, r in result.basic.items()
        },
        "two_step": None,
    }
    if result.two_step is not None:
        x = result.two_step.x_tod
        summary["two_step"] = {}
        for name, it in (("first_estimate", result.two_step.iter1), ("second_estimate", result.two_step.iter2)):
            summary["two_step"][name] = {
                "noise_status": it.noise_status, "noise_n_valid": it.noise_n_valid,
                "explained_ratio": [float(v) for v in it.explained_ratio],
                "max_abs_identity_err_k": float(np.max(np.abs(x - (it.c_tod + it.a_tod + it.e_tod)))),
                **_evaluate_spectrum(it.restored_spectrum, it.hits, data, truth, line_def),
            }
    return summary


def check_composition(data: ObservedData, truth: ObservationTruth) -> dict:
    """observed = signal_tod + Σ noise が保存データで成り立つか確認する(評価段階の整合性確認)

    生成時と同じ順序(0 に雑音成分を順に足し、最後に天体信号と足す)で再計算するので、
    float64 無損失保存なら差は厳密に 0 になる。
    """
    total = np.zeros(data.observed.shape)
    for tod in truth.noise.values():
        total += tod
    err = float(np.max(np.abs(data.observed - (truth.signal_tod + total))))
    return {"max_abs_composition_err_k": err, "is_exact": err == 0.0}
