"""段階 2-A:輝線幅・変調幅・除去法(simple_pca / iterative_pca)のパラメータスイープ

1 点観測(段階 1 と同じ観測)で、config の sweep セクションに書いた値の全組み合わせを実行し、
評価値を CSV / JSON に保存して比較図を出力する。

使い方:
    python run_step2_sweep.py
    python run_step2_sweep.py --fwhm 50 100 200 400 --half-width 150 300
    python run_step2_sweep.py --set noise.atmosphere.offset_k=0 --outdir outputs/step2_sweep_offset0

乱数は seed から成分ごとに作るので、どの条件でも熱雑音・大気の乱数列は同じ
(条件の違いだけを比べられる)。
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from fmlo_sim import evaluation, pipeline, plotting
from fmlo_sim.config import config_from_dict, parse_set_args, read_yaml
from fmlo_sim.provenance import collect_provenance
from run_fig6 import _get_git_info

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "step2_sweep.yaml"

CSV_KEYS = [
    "line_fwhm_kms", "half_width_ch", "method", "line_fwhm_ch", "mod_full_width_ch",
    "line_fwhm_over_mod_width", "peak_ratio", "integrated_ratio", "rmse_k",
    "line_free_residual_rms_k", "line_free_residual_mean_k", "expected_thermal_rms_k",
    "n_fully_observed_channels", "line_min_hits", "line_min_hits_frac", "band_edge_hits",
    "n_iter", "converged", "final_rel_change", "final_model_norm_k",
]


def condition_metrics(cfg, red, sim) -> dict:
    """1 条件の評価値(evaluation.evaluate に RMSE とヒット数の情報を足す)"""
    grid = sim.grid
    fwhm_ghz = cfg.source.center_freq_ghz * cfg.source.line_fwhm_kms / 299792.458
    fwhm_ch = fwhm_ghz / grid.channel_width_ghz
    full_width = 2 * cfg.modulation.half_width_ch
    in_line, _ = evaluation.line_masks(grid.freq_ghz, cfg.source.center_freq_ghz,
                                       cfg.source.line_fwhm_kms, cfg.evaluation.line_free_n_fwhm)
    resid = evaluation.residual_summary(red.restored_k, sim.truth_k, red.hits, cfg.n_time)
    observed = red.hits > 0
    line_hits = red.hits[in_line]
    return {
        "line_fwhm_kms": cfg.source.line_fwhm_kms,
        "half_width_ch": cfg.modulation.half_width_ch,
        "method": cfg.removal.method,
        "line_fwhm_ch": float(fwhm_ch),
        "mod_full_width_ch": full_width,
        "line_fwhm_over_mod_width": float(fwhm_ch / full_width) if full_width else None,
        "rmse_k": resid["residual_rms_k"],
        "n_fully_observed_channels": resid["n_fully_observed_channels"],
        # 輝線領域(±n_fwhm × FWHM)の中で最も少ないヒット数。n_time 未満なら輝線の一部が帯域端に掛かる
        "line_min_hits": int(line_hits.min()) if line_hits.size else None,
        "line_min_hits_frac": float(line_hits.min() / cfg.n_time) if line_hits.size else None,
        # 観測された周波数範囲の両端のヒット数(変調の端では 1 回しか見ていない)
        "band_edge_hits": [int(red.hits[observed][0]), int(red.hits[observed][-1])],
        "expected_thermal_rms_k": pipeline.expected_thermal_rms_k(cfg),
        **red.metrics,
        "n_iter": red.metrics.get("n_iter", 1),
        "converged": red.metrics.get("converged"),
        "final_rel_change": red.metrics.get("final_rel_change"),
        "final_model_norm_k": red.metrics.get("final_model_norm_k"),
    }


def run_sweep(base: dict, sweep: dict, overrides: dict, outdir: Path, make_plots=True):
    keys = list(sweep)
    rows, details = [], []
    for values in itertools.product(*(sweep[k] for k in keys)):
        cond = dict(zip(keys, values))
        cfg = config_from_dict(base, {**overrides, **cond})
        sim = pipeline.simulate(cfg)
        red = pipeline.reduce_by_method(sim)
        row = condition_metrics(cfg, red, sim)
        rows.append(row)
        details.append({"condition": cond, "metrics": row})
        print(f"  FWHM {cfg.source.line_fwhm_kms:6.1f} km/s  half width {cfg.modulation.half_width_ch:4d} ch  "
              f"{cfg.removal.method:<14} 積分強度比 {row['integrated_ratio']:.4f}  "
              f"残差RMS {row['line_free_residual_rms_k']:.4f} K  反復 {row['n_iter']}")
    return rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="設定ファイル(既定: config/step2_sweep.yaml)")
    parser.add_argument("--outdir", default=str(HERE / "outputs" / "step2_sweep"))
    parser.add_argument("--fwhm", type=float, nargs="+", default=None, help="輝線幅 [km/s] のリスト")
    parser.add_argument("--half-width", type=int, nargs="+", default=None, help="変調の半幅 [ch] のリスト")
    parser.add_argument("--methods", nargs="+", default=None, choices=["simple_pca", "iterative_pca"])
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="全条件に共通の上書き")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data = read_yaml(args.config)
    sweep = dict(data.pop("sweep", {}) or {})
    if args.fwhm:
        sweep["source.line_fwhm_kms"] = args.fwhm
    if args.half_width:
        sweep["modulation.half_width_ch"] = args.half_width
    if args.methods:
        sweep["removal.method"] = args.methods
    if not sweep:
        raise SystemExit("スイープする値がありません(config の sweep セクションか --fwhm などで指定)")
    overrides = parse_set_args(args.set)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"スイープ: {sweep}")
    rows, details = run_sweep(data, sweep, overrides, outdir)

    with open(outdir / "sweep_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_KEYS)
        for r in rows:
            writer.writerow([json.dumps(r[k]) if isinstance(r[k], list) else r[k] for k in CSV_KEYS])
    base_cfg = config_from_dict(data, overrides)
    with open(outdir / "sweep_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"base_config": base_cfg.to_dict(), "sweep": sweep, "overrides": overrides,
                   "provenance": collect_provenance(_get_git_info(), sys.argv, str(args.config)),
                   "conditions": details},
                  f, indent=2, ensure_ascii=False, default=float)
    if not args.no_plots:
        plotting.plot_sweep(outdir / "sweep_comparison.png", rows, pipeline.expected_thermal_rms_k(base_cfg))
    print(f"出力先: {outdir}")


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
