"""段階1「1ビーム・1点観測」FMLO シミュレーションの実行スクリプト

使い方:
    python run_step1.py
    python run_step1.py --outdir outputs/offset50 --set noise.atmosphere.offset_k=50
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


def run(cfg, outdir, make_plots=True, save_cube=True):
    """1回分のシミュレーションを実行し、評価値の辞書を返す

    PCA の平均の扱い(none / channel_mean)と PCA なしの3通りを、同じ観測データで評価する。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim = pipeline.simulate(cfg)
    results = pipeline.reduce_all_modes(sim)
    grid = sim.grid

    # 輝線が一度も入らないチャンネルで、時間方向のパワースペクトルを PCA 前後で比べる
    _, line_free_wide = evaluation.line_masks(
        grid.freq_ghz, cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm,
    )
    ch_mask = evaluation.line_free_channels(line_free_wide, sim.shifts_ch, grid.start0, grid.n_channels)
    dt = cfg.modulation.dt_s
    psd = {"before": evaluation.time_power_spectrum(sim.observed, dt, ch_mask)}
    for mode in MEAN_MODES:
        psd[mode] = evaluation.time_power_spectrum(results[mode].pca.cleaned, dt, ch_mask)

    summary = {
        "config": cfg.to_dict(),
        "primary_mean_mode": cfg.removal.mean_mode,
        "n_time": cfg.n_time,
        "expected_thermal_rms_k": pipeline.expected_thermal_rms_k(cfg),
        "results": {
            label: {"n_components": r.n_components, "mean_mode": r.mean_mode,
                    "explained_ratio": [float(x) for x in r.pca.explained_ratio], **r.metrics}
            for label, r in results.items()
        },
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
        hits=results["none"].hits,
    )
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(outdir / "metrics_comparison.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["n_components", "mean_mode", *next(iter(results.values())).metrics.keys()]
        writer = csv.writer(f)
        writer.writerow(["label", *keys])
        for label, r in summary["results"].items():
            writer.writerow([label, *[r[k] for k in keys]])

    if make_plots:
        primary = results[cfg.removal.mean_mode]
        plotting.plot_cube(outdir / "fig01_cube.png", sim)
        plotting.plot_shift_pattern(outdir / "fig02_shift_pattern.png", sim)
        plotting.plot_modulated_signal(outdir / "fig03_modulated_signal.png", sim)
        plotting.plot_noise(outdir / "fig04_noise_components.png", sim)
        plotting.plot_pca(outdir / "fig05_pca_removal.png", sim, primary)
        plotting.plot_restored(outdir / "fig06_restored_spectrum.png", sim, results)
        plotting.plot_power_spectra(outdir / "fig07_power_spectra.png", psd)

    return summary


def print_summary(summary):
    print(f"熱雑音だけを時間平均したときの期待 RMS: {summary['expected_thermal_rms_k']:.4f} K")
    print(f"{'label':<14}{'残差RMS [K]':>12}{'ピーク比':>10}{'積分強度比':>10}")
    for label, r in summary["results"].items():
        print(f"{label:<14}{r['line_free_residual_rms_k']:>14.4f}"
              f"{r['peak_ratio']:>13.4f}{r['integrated_ratio']:>14.4f}")


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

    cfg = load_config(args.config, overrides)
    summary = run(cfg, args.outdir, make_plots=not args.no_plots)
    print_summary(summary)
    print(f"出力先: {args.outdir}")


if __name__ == "__main__":
    main()
