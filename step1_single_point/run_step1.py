"""段階1「1ビーム・1点観測」FMLO シミュレーションの実行スクリプト

使い方:
    python run_step1.py
    python run_step1.py --outdir outputs/offset50 --set noise.atmosphere.offset_k=50
    python run_step1.py --fa-factors 3 --fa-remove 0,1
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fmlo_sim import cube as cube_mod
from fmlo_sim import evaluation, pipeline, plotting
from fmlo_sim.config import MEAN_MODES, load_config, parse_set_args

HERE = Path(__file__).resolve().parent


def run(cfg, outdir, make_plots=True, save_cube=True, only_method="all"):
    """1回分のシミュレーションを実行し、評価値の辞書を返す

    - none: PCA (mean_mode='none')
    - channel_mean: PCA (mean_mode='channel_mean')
    - no_pca: 除去なし
    - fa_ica: FA+ICA
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim = pipeline.simulate(cfg)
    all_results = pipeline.reduce_all_modes(sim)

    # フィルタリング (only_method)
    if only_method == "pca":
        results = {k: v for k, v in all_results.items() if k in ("none", "channel_mean", "no_pca")}
    elif only_method == "fa_ica":
        results = {k: v for k, v in all_results.items() if k in ("fa_ica", "no_pca")}
    else:
        results = all_results

    grid = sim.grid

    # 輝線が一度も入らないチャンネルで、時間方向のパワースペクトルを除去前後で比べる
    _, line_free_wide = evaluation.line_masks(
        grid.freq_ghz, cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm,
    )
    ch_mask = evaluation.line_free_channels(line_free_wide, sim.shifts_ch, grid.start0, grid.n_channels)
    dt = cfg.modulation.dt_s
    psd = {"before": evaluation.time_power_spectrum(sim.observed, dt, ch_mask)}
    for mode in MEAN_MODES:
        if mode in results:
            psd[mode] = evaluation.time_power_spectrum(results[mode].cleaned, dt, ch_mask)
    if "fa_ica" in results:
        psd["fa_ica"] = evaluation.time_power_spectrum(results["fa_ica"].cleaned, dt, ch_mask)

    results_summary = {}
    for label, r in results.items():
        item = {
            "method": r.method,
            "n_components": r.n_components,
            "mean_mode": r.mean_mode,
            **r.metrics,
        }
        if r.pca is not None:
            item["explained_ratio"] = [float(x) for x in r.pca.explained_ratio]
        elif r.fa_ica is not None:
            item["fa_ica_diag"] = {
                "n_factors": r.fa_ica.n_factors,
                "remove_indices": r.fa_ica.remove_indices,
                "fa_n_iter": r.fa_ica.fa_n_iter,
                "ica_n_iter": r.fa_ica.ica_n_iter,
            }
        results_summary[label] = item

    summary = {
        "config": cfg.to_dict(),
        "primary_mean_mode": cfg.removal.mean_mode,
        "n_time": cfg.n_time,
        "expected_thermal_rms_k": pipeline.expected_thermal_rms_k(cfg),
        "results": results_summary,
    }

    if save_cube:
        cube_mod.save_cube(outdir / "cube.npz", sim.cube, sim.ra_arcsec, sim.dec_arcsec,
                           grid.freq_ghz, cfg)
    
    np.savez(
        outdir / "timestream.npz",
        time_s=sim.time_s, shifts_ch=sim.shifts_ch, freq_wide_ghz=grid.freq_ghz,
        start0=grid.start0, truth_k=sim.truth_k, signal_tod=sim.signal_tod.astype(np.float32),
        observed=sim.observed.astype(np.float32),
        **{f"noise_{name}": tod.astype(np.float32) for name, tod in sim.noise.components.items()},
        **{f"restored_{label}_k": r.restored_k for label, r in results.items()},
        hits=results["no_pca"].hits if "no_pca" in results else next(iter(results.values())).hits,
    )
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(outdir / "metrics_comparison.csv", "w", newline="", encoding="utf-8") as f:
        metric_keys = list(next(iter(results.values())).metrics.keys())
        keys = ["method", "mean_mode", "n_components", *metric_keys]
        writer = csv.writer(f)
        writer.writerow(["label", *keys])
        for label, r in results_summary.items():
            writer.writerow([label, *[r.get(k, "") for k in keys]])

    if make_plots:
        plotting.plot_cube(outdir / "fig01_cube.png", sim)
        plotting.plot_shift_pattern(outdir / "fig02_shift_pattern.png", sim)
        plotting.plot_modulated_signal(outdir / "fig03_modulated_signal.png", sim)
        plotting.plot_noise(outdir / "fig04_noise_components.png", sim)
        if cfg.removal.mean_mode in results and results[cfg.removal.mean_mode].pca is not None:
            plotting.plot_pca(outdir / "fig05_pca_removal.png", sim, results[cfg.removal.mean_mode])
        if "fa_ica" in results and results["fa_ica"].fa_ica is not None:
            plotting.plot_fa_ica_components(outdir / "fig05_fa_ica_components.png", sim, results["fa_ica"])
        plotting.plot_restored(outdir / "fig06_restored_spectrum.png", sim, results)
        plotting.plot_power_spectra(outdir / "fig07_power_spectra.png", psd)

    return summary


def print_summary(summary):
    print(f"熱雑音だけを時間平均したときの期待 RMS: {summary['expected_thermal_rms_k']:.4f} K")
    print(f"{'label':<16}{'method':<10}{'mean_mode':<14}{'残差RMS [K]':>12}{'ピーク比':>10}{'積分強度比':>10}")
    for label, r in summary["results"].items():
        print(f"{label:<16}{r.get('method', ''):<10}{r.get('mean_mode', ''):<14}"
              f"{r['line_free_residual_rms_k']:>12.4f}"
              f"{r['peak_ratio']:>10.4f}{r['integrated_ratio']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="設定ファイル(既定: config/default.yaml)")
    parser.add_argument("--outdir", default=str(HERE / "outputs" / "step1"), help="出力先ディレクトリ")
    parser.add_argument("--half-width-ch", "--mod-width", dest="half_width_ch", type=int, default=None,
                        help="変調の半幅 [ch](既定値: configの値)")
    parser.add_argument("--zigzag-step-ch", "--mod-step", dest="zigzag_step_ch", type=int, default=None,
                        help="ジグザグの1サンプルあたりのシフト量 [ch](既定値: configの値)")
    parser.add_argument("--atmosphere-model", "--noise-type", dest="atmosphere_model",
                        choices=["laminar", "turbulent", "colored"], default=None,
                        help="大気雑音モデル ('laminar': 層流, 'turbulent'/'colored': 乱流、既定値: configの値)")
    parser.add_argument("--method", choices=["all", "pca", "fa_ica"], default="all",
                        help="実行する手法 ('all', 'pca', 'fa_ica', 既定: all)")
    parser.add_argument("--fa-factors", type=int, default=None, help="FA+ICA の共通因子数")
    parser.add_argument("--fa-remove", type=str, default=None,
                        help="除去する ICA 成分インデックス(カンマ区切り、例: '0' または '0,1')")
    parser.add_argument("--fa-mean-mode", choices=["channel_mean", "none"], default=None,
                        help="FA+ICA の平均モード ('channel_mean' または 'none')")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="設定の上書き(例: --set removal.n_components=2)。複数指定可")
    parser.add_argument("--no-plots", action="store_true", help="図を出力しない")
    args = parser.parse_args()

    overrides = parse_set_args(args.set)
    if args.half_width_ch is not None:
        overrides["modulation.half_width_ch"] = args.half_width_ch
    if args.zigzag_step_ch is not None:
        overrides["modulation.zigzag_step_ch"] = args.zigzag_step_ch
    if args.atmosphere_model is not None:
        overrides["noise.atmosphere.model"] = args.atmosphere_model
    if args.fa_factors is not None:
        overrides["removal.fa_ica.n_factors"] = args.fa_factors
    if args.fa_remove is not None:
        overrides["removal.fa_ica.remove_indices"] = [int(x.strip()) for x in args.fa_remove.split(",") if x.strip()]
    if args.fa_mean_mode is not None:
        overrides["removal.fa_ica.mean_mode"] = args.fa_mean_mode

    cfg = load_config(args.config, overrides)
    summary = run(cfg, args.outdir, make_plots=not args.no_plots, only_method=args.method)
    print_summary(summary)
    print(f"出力先: {args.outdir}")


if __name__ == "__main__":
    main()
