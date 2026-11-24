"""B. 保存した模擬観測データ(run_observe_cube.py の出力)を読み込み、解析・評価する

推定(PCA 除去・復調、任意で 2 回推定)には観測時系列・シフト量・軸だけを使い、
正解スペクトルは評価でだけ参照する。

使い方:
    # 内部生成キューブ由来(生成条件が記録されている)なら輝線領域を自動で決められる
    python run_analyze_observation.py --obs outputs/observe_demo/observation.npz \
        --outdir outputs/observe_demo/analysis --line-from-source

    # 外部キューブでは輝線領域を周波数窓で明示する(複数可)
    python run_analyze_observation.py --obs obs.npz --outdir out \
        --line-window-ghz 149.95 150.05 --ref-freq-ghz 150.0

    # 2 回推定も行う(run_fig6.py と同じ処理)
    python run_analyze_observation.py --obs obs.npz --outdir out --line-from-source --two-step

解析条件(除去成分数・平均の扱い・輝線領域の n_fwhm)は設定 YAML の removal / evaluation
セクションから読む(既定: config/default.yaml。--set removal.n_components=2 などで上書き)。
既存の出力ファイルは上書きしない(--overwrite を付けた場合だけ上書き)。

出力:
    analysis.npz  復元スペクトル・ヒット数・周波数軸(float64)。2 回推定時はその推定スペクトルも
    metrics.json  解析条件・評価結果・観測データの識別情報・来歴
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fmlo_sim import analysis, obsio
from fmlo_sim.config import DEFAULT_CONFIG_PATH, parse_set_args
from fmlo_sim.provenance import collect_provenance
from run_fig6 import _get_git_info

HERE = Path(__file__).resolve().parent


def analyze_and_save(obs_path, outdir, config_path=None, overrides=None, line_def=None,
                     line_from_source=False, two_step=False, cutoff_sigma=5.0,
                     overwrite=False, cli_argv=None):
    """観測データを読んで解析・評価し、結果を保存する。評価結果の辞書を返す"""
    outdir = Path(outdir)
    config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    out_npz, out_json = outdir / "analysis.npz", outdir / "metrics.json"
    obsio.refuse_overwrite([out_npz, out_json], overwrite)

    acfg = analysis.load_analysis_config(config_path, overrides)
    data, truth = obsio.load_observation(obs_path)
    if line_from_source:
        if line_def is not None:
            raise ValueError("輝線領域の指定は --line-from-source と --line-window-ghz のどちらか一方です")
        line_def = analysis.line_definition_from_source(data.meta, acfg.evaluation.line_free_n_fwhm)

    # 推定: 観測データだけを渡す(truth は渡さない)
    result = analysis.analyze(data, acfg.removal.n_components, acfg.removal.mean_mode,
                              two_step=two_step, cutoff_sigma=cutoff_sigma if two_step else None)
    # 評価: ここで初めて正解を参照する
    metrics = analysis.evaluate(result, data, truth, line_def)

    meta = data.meta
    summary = {
        "observation": {
            "path": data.path, "file_sha256": data.file_sha256,
            "cube": meta["cube"], "pointing": meta["pointing"], "grid": meta["grid"],
            "n_time": meta["n_time"], "seed": meta["seed"], "setup": meta["setup"],
            "noise_components": meta["noise_components"],
            "generated_by": meta["provenance"],
        },
        "analysis_conditions": {
            "n_components": acfg.removal.n_components,
            "mean_mode_for_two_step": acfg.removal.mean_mode,
            "basic_labels": list(result.basic),
            "two_step": bool(two_step),
            "cutoff_sigma": float(cutoff_sigma) if two_step else None,
            "evaluation_region": "全時刻で観測された周波数(hits == n_time)",
            "line_definition": None if line_def is None else line_def.to_dict(),
            "truth_used_in_estimation": False,
        },
        "composition_check": analysis.check_composition(data, truth),
        "results": metrics,
        "precision": {"analysis_npz_float_arrays": "float64(丸めなし)"},
        "provenance": {
            **collect_provenance(_get_git_info(), cli_argv, str(config_path)),
            "config_overrides": overrides or {},
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "freq_wide_ghz": data.freq_wide_ghz,
        "hits": result.basic["none"].hits,
        **{f"restored_{label}_k": r.restored_k for label, r in result.basic.items()},
    }
    if result.two_step is not None:
        for i, it in enumerate((result.two_step.iter1, result.two_step.iter2), start=1):
            arrays[f"two_step_restored{i}_k"] = it.restored_spectrum
            arrays[f"two_step_model{i}_k"] = it.model_spectrum
            arrays[f"two_step_noise_sigma{i}_k"] = it.noise_sigma
    np.savez(out_npz, **arrays)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _fmt(value, spec):
    return "未計算" if value is None else format(value, spec)


def print_summary(summary):
    print(f"合成の確認 observed = signal + Σnoise: 最大差 {summary['composition_check']['max_abs_composition_err_k']:.3g} K")
    print(f"{'label':<16}{'全域残差RMS [K]':>16}{'輝線外残差RMS [K]':>18}{'ピーク比':>10}{'積分強度比':>10}")
    rows = dict(summary["results"]["basic"])
    for name, r in (summary["results"]["two_step"] or {}).items():
        rows[f"two_step/{name.split('_')[0]}"] = r
    for label, r in rows.items():
        line = r["line_metrics"] or {}
        print(f"{label:<16}{_fmt(r['residual_rms_k'], '>18.4f')}"
              f"{_fmt(line.get('line_free_residual_rms_k'), '>20.4f')}"
              f"{_fmt(line.get('peak_ratio'), '>13.4f')}{_fmt(line.get('integrated_ratio'), '>14.4f')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs", required=True, help="観測データ(.npz、run_observe_cube.py の出力)")
    parser.add_argument("--outdir", required=True, help="出力先ディレクトリ")
    parser.add_argument("--config", default=None,
                        help="解析設定 YAML(removal / evaluation を使う。既定: config/default.yaml)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="解析設定の上書き(例: --set removal.n_components=2)。複数指定可")
    parser.add_argument("--line-from-source", action="store_true",
                        help="キューブに記録されたガウシアン天体の生成条件から輝線領域を決める")
    parser.add_argument("--line-window-ghz", action="append", nargs=2, type=float, metavar=("LO", "HI"),
                        help="輝線領域の周波数窓 [GHz]。複数指定可")
    parser.add_argument("--ref-freq-ghz", type=float, default=None,
                        help="積分強度の速度換算に使う基準周波数 [GHz](--line-window-ghz と併用、必須)")
    parser.add_argument("--two-step", action="store_true", help="2 回推定(run_fig6.py と同じ処理)も行う")
    parser.add_argument("--cutoff-sigma", type=float, default=5.0, help="2 回推定の sigma-cutoff 閾値")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力ファイルを上書きする")
    args = parser.parse_args()

    line_def = None
    if args.line_window_ghz:
        if args.ref_freq_ghz is None:
            parser.error("--line-window-ghz を使う場合は --ref-freq-ghz も指定してください")
        line_def = analysis.LineDefinition(
            "windows", windows_ghz=[tuple(w) for w in args.line_window_ghz],
            ref_freq_ghz=args.ref_freq_ghz, origin="コマンドライン (--line-window-ghz)",
        )
    elif args.ref_freq_ghz is not None:
        parser.error("--ref-freq-ghz は --line-window-ghz と併せて指定してください")

    summary = analyze_and_save(
        args.obs, args.outdir, config_path=args.config, overrides=parse_set_args(args.set),
        line_def=line_def, line_from_source=args.line_from_source, two_step=args.two_step,
        cutoff_sigma=args.cutoff_sigma, overwrite=args.overwrite, cli_argv=sys.argv,
    )
    print_summary(summary)
    print(f"出力先: {args.outdir}")


if __name__ == "__main__":
    main()
