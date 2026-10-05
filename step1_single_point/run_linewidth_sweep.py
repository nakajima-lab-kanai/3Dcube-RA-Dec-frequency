"""輝線幅(と変調幅)を変えて一括実行するスクリプト

使い方:
    python run_linewidth_sweep.py
    python run_linewidth_sweep.py --line-fwhm-kms 50 100 200 --half-width-ch 150 300 600
"""
import argparse
import csv
from pathlib import Path

from fmlo_sim.config import load_config, parse_set_args
from run_step1 import run

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="設定ファイル(既定: config/default.yaml)")
    parser.add_argument("--outdir", default=str(HERE / "outputs" / "linewidth_sweep"))
    parser.add_argument("--line-fwhm-kms", type=float, nargs="+", default=[50.0, 100.0, 200.0],
                        help="輝線幅 FWHM [km/s] のリスト")
    parser.add_argument("--half-width-ch", type=int, nargs="+", default=None,
                        help="変調の半幅 [ch] のリスト(省略時は設定ファイルの値)")
    parser.add_argument("--zigzag-step-ch", type=int, default=None,
                        help="ジグザグの1サンプルあたりのシフト量 [ch](省略時は設定ファイルの値)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="その他の設定の上書き")
    parser.add_argument("--no-plots", action="store_true", help="図を出力しない")
    args = parser.parse_args()

    base = parse_set_args(args.set)
    if args.zigzag_step_ch is not None:
        base["modulation.zigzag_step_ch"] = args.zigzag_step_ch
    half_widths = args.half_width_ch or [load_config(args.config, base).modulation.half_width_ch]

    outdir = Path(args.outdir)
    rows = []
    for fwhm in args.line_fwhm_kms:
        for half_width in half_widths:
            cfg = load_config(args.config, {
                **base, "source.line_fwhm_kms": fwhm, "modulation.half_width_ch": half_width,
            })
            name = f"fwhm{fwhm:g}kms_hw{half_width}ch"
            print(f"--- {name} ---")
            # キューブは設定から再生成できるので、一括実行では保存しない
            summary = run(cfg, outdir / name, make_plots=not args.no_plots, save_cube=False)
            for label, r in summary["results"].items():
                rows.append({
                    "line_fwhm_kms": fwhm, "half_width_ch": half_width, "label": label,
                    "n_components": r["n_components"],
                    "line_free_residual_rms_k": r["line_free_residual_rms_k"],
                    "peak_ratio": r["peak_ratio"], "integrated_ratio": r["integrated_ratio"],
                })
                print(f"  {label:<14} 残差RMS = {r['line_free_residual_rms_k']:.4f} K, "
                      f"ピーク比 = {r['peak_ratio']:.4f}, 積分強度比 = {r['integrated_ratio']:.4f}")

    with open(outdir / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"まとめ: {outdir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
