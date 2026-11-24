"""段階 2-B:1 ビーム OTF + FMLO 空間スキャンのシミュレーション・復元・評価

正解キューブ(ビーム畳み込み)→ スキャン軌跡に沿った FMLO 観測(x_scan / y_scan)→
雑音除去(simple_pca / iterative_pca)→ グリッディング → basket weave 統合 → 評価・図。

使い方:
    python run_step2_scan.py
    python run_step2_scan.py --set removal.method=simple_pca --outdir outputs/step2_scan_simple
    python run_step2_scan.py --set scan.mode=x_scan
    python run_step2_scan.py --joint-basket-weave    # x と y をまとめて反復(輝線モデルを共有)

評価の基準(reference)は「天体信号だけの時系列を同じ手順でグリッディングしたキューブ」。
グリッディングの平滑化は基準にも同じだけ入るので、残差は雑音と雑音除去による誤差だけになる。
正解キューブ(ビーム畳み込み済み)を出力グリッドに補間した truth との比較も併記する。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from fmlo_sim import cube as cube_mod
from fmlo_sim import evaluation, mapping, modulation, noise, pipeline, plotting, scan
from fmlo_sim.config import load_config, parse_set_args
from fmlo_sim.provenance import collect_provenance
from run_fig6 import _get_git_info

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "step2_scan.yaml"

CSV_KEYS = [
    "cube_rmse_vs_reference_k", "cube_rmse_vs_truth_k", "cube_line_free_rms_k",
    "voxel_peak_ratio", "mom0_peak_ratio", "mom0_residual_rms_kkms", "peak_position_error_arcsec",
    "centroid_error_arcsec", "flux_recovery_vs_reference", "flux_recovery_vs_truth",
    "n_iter", "converged", "n_eval_pixels",
]


def full_coverage_freq(shifts_list, start0, n_channels, n_wide):
    """全パス・全サンプルで観測された周波数 (n_wide,)"""
    ok = np.zeros(n_wide, dtype=bool)
    lo = start0 + max(int(np.max(s)) for s in shifts_list)
    hi = start0 + min(int(np.min(s)) for s in shifts_list) + n_channels
    ok[lo:hi] = True
    return ok


def scanned_pixels(map_grid, passes):
    """出力画素のうち、パスの走査範囲(外接矩形)の内側にある画素 (n_ra, n_dec)"""
    ra = np.concatenate([p.ra_arcsec for p in passes])
    dec = np.concatenate([p.dec_arcsec for p in passes])
    tol = 1e-6
    in_ra = (map_grid.ra_arcsec >= ra.min() - tol) & (map_grid.ra_arcsec <= ra.max() + tol)
    in_dec = (map_grid.dec_arcsec >= dec.min() - tol) & (map_grid.dec_arcsec <= dec.max() + tol)
    return in_ra[:, None] & in_dec[None, :]


def run(cfg, outdir, make_plots=True, joint=False, cli_argv=None, config_path=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = cfg.source

    # 1. 正解キューブ(ビーム畳み込み)とスキャン軌跡
    sky = pipeline.make_internal_sky_cube(cfg)
    passes = scan.make_passes(cfg)
    map_grid = mapping.make_map_grid(cfg)

    # 2. 各パスの模擬観測(パスごとに独立な乱数)
    obs = [scan.observe_pass(sky, cfg, p, i) for i, p in enumerate(passes)]
    grid = obs[0].grid
    start0, n_wide, n_ch = grid.start0, grid.n_wide, grid.n_channels
    m = cfg.mapping
    geoms = [mapping.PassGeometry(p.name, p.ra_arcsec, p.dec_arcsec, o.shifts_ch,
                                  mapping.gridding_matrix(map_grid, p.ra_arcsec, p.dec_arcsec,
                                                          m.kernel_fwhm_arcsec, m.kernel_support_fwhm))
             for p, o in zip(passes, obs)]

    # 3. 雑音除去 → グリッディング(パスごと。正解は使わない)
    reds = {o.scan_pass.name: mapping.reduce_passes(o.scan_pass.name, [o.observed], [g], map_grid,
                                                    start0, n_wide, cfg.removal)
            for o, g in zip(obs, geoms)}
    refs = {o.scan_pass.name: mapping.grid_tod(o.signal_tod, g.shifts_ch, start0, n_wide, g.wmat, map_grid.shape)
            for o, g in zip(obs, geoms)}
    cubes = {name: r.gridded.data for name, r in reds.items()}
    ref_cubes = {name: r.data for name, r in refs.items()}
    label_passes = {p.name: [p] for p in passes}

    # 4. basket weave 統合
    if cfg.scan.mode == "basket_weave":
        rx, ry = reds["x_scan"], reds["y_scan"]
        if m.basket_weave_method == "weighted_mean":
            cubes["basket_weave"] = mapping.combine_weighted_mean([rx.gridded, ry.gridded]).data
            ref_cubes["basket_weave"] = mapping.combine_weighted_mean([refs["x_scan"], refs["y_scan"]]).data
        else:
            cubes["basket_weave"] = mapping.combine_fourier(cubes["x_scan"], cubes["y_scan"])
            ref_cubes["basket_weave"] = mapping.combine_fourier(ref_cubes["x_scan"], ref_cubes["y_scan"])
        label_passes["basket_weave"] = passes
        if joint:
            jr = mapping.reduce_passes("basket_weave_joint", [o.observed for o in obs], geoms, map_grid,
                                       start0, n_wide, cfg.removal)
            reds["basket_weave_joint"] = jr
            cubes["basket_weave_joint"] = jr.gridded.data
            ref_cubes["basket_weave_joint"] = ref_cubes["basket_weave"] if m.basket_weave_method == "weighted_mean" \
                else mapping.combine_weighted_mean([refs["x_scan"], refs["y_scan"]]).data
            label_passes["basket_weave_joint"] = passes

    # 5. 評価
    rr, dd = map_grid.pixel_centers()
    truth_grid = cube_mod.sample_spectra(sky.data, sky.ra_arcsec, sky.dec_arcsec, rr, dd).reshape(
        *map_grid.shape, n_wide)
    in_line, line_free = evaluation.line_masks(grid.freq_ghz, src.center_freq_ghz, src.line_fwhm_kms,
                                               cfg.evaluation.line_free_n_fwhm)
    shifts_by_name = {o.scan_pass.name: o.shifts_ch for o in obs}
    metrics, eval_masks = {}, {}
    for label, cube in cubes.items():
        ps = label_passes[label]
        freq_valid = full_coverage_freq([shifts_by_name[p.name] for p in ps], start0, n_ch, n_wide)
        spatial = scanned_pixels(map_grid, ps)
        eval_masks[label] = (spatial, freq_valid)
        mm = evaluation.evaluate_scan_cube(
            cube, ref_cubes[label], truth_grid, spatial, freq_valid, in_line, line_free,
            map_grid.ra_arcsec, map_grid.dec_arcsec, grid.channel_width_ghz, src.center_freq_ghz,
            map_grid.pixel_arcsec, src.center_ra_arcsec, src.center_dec_arcsec)
        red = reds.get(label)
        it = red.iterative if red is not None else None
        if it is not None:
            mm.update({"n_iter": it.n_iter, "converged": it.converged, "rel_change": it.rel_change,
                       "final_model_norm_k": it.model_norm[-1]})
        elif label == "basket_weave":
            mm.update({"n_iter": [reds["x_scan"].iterative.n_iter, reds["y_scan"].iterative.n_iter]
                       if reds["x_scan"].iterative else 1,
                       "converged": [reds["x_scan"].iterative.converged, reds["y_scan"].iterative.converged]
                       if reds["x_scan"].iterative else None})
        else:
            mm.update({"n_iter": 1, "converged": None})
        metrics[label] = mm

    # 除去前後の PSD(輝線が一度も入らないチャンネル)と理論熱雑音レベル
    sigma_t = noise.radiometer_sigma_k(cfg.noise.thermal.tsys_k, cfg.spectrometer.channel_width_mhz * 1e6,
                                       cfg.modulation.dt_s) if cfg.noise.thermal.enabled else 0.0
    psd = {}
    for o in obs:
        name = o.scan_pass.name
        ch_mask = evaluation.line_free_channels(line_free, o.shifts_ch, start0, n_ch)
        psd[name] = evaluation.psd_comparison(
            {"before removal": o.observed, f"after {cfg.removal.method}": reds[name].cleaned[0]},
            cfg.modulation.dt_s, ch_mask, sigma_t)

    summary = {
        "config": cfg.to_dict(),
        "sampling": scan.sampling_summary(cfg, passes),
        "map_grid": {"n_ra": map_grid.shape[0], "n_dec": map_grid.shape[1],
                     "pixel_arcsec": map_grid.pixel_arcsec},
        "thermal_sigma_per_sample_k": float(sigma_t),
        "metrics": metrics,
        "psd_excess": {name: c["excess"] for name, c in psd.items()},
        "provenance": collect_provenance(_get_git_info(), cli_argv, config_path),
    }
    with open(outdir / "metrics_scan.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    with open(outdir / "metrics_scan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", *CSV_KEYS])
        for label, mm in metrics.items():
            w.writerow([label, *[json.dumps(mm.get(k)) if isinstance(mm.get(k), list) else mm.get(k)
                                 for k in CSV_KEYS]])
    np.savez_compressed(
        outdir / "cubes_scan.npz",
        ra_arcsec=map_grid.ra_arcsec, dec_arcsec=map_grid.dec_arcsec, freq_ghz=grid.freq_ghz,
        truth_on_grid=truth_grid.astype(np.float32),
        **{f"restored_{k}": v.astype(np.float32) for k, v in cubes.items()},
        **{f"reference_{k}": v.astype(np.float32) for k, v in ref_cubes.items()},
    )

    if make_plots:
        dv_line = in_line
        truth_mom_sky = evaluation.moment0_map(sky.data, dv_line, grid.channel_width_ghz, src.center_freq_ghz)
        plotting.plot_scan_trajectory(outdir / "fig21_scan_trajectory.png", passes, truth_mom_sky,
                                      sky.ra_arcsec, sky.dec_arcsec, cfg.telescope.beam_fwhm_arcsec)
        o0, r0 = obs[0], reds[obs[0].scan_pass.name]
        demod, _ = modulation.demodulate_tod(r0.cleaned[0], o0.shifts_ch, start0, n_wide)
        plotting.plot_scan_timestreams(outdir / "fig22_scan_timestreams.png", o0, r0.cleaned[0], demod,
                                       grid.freq_ghz)
        del demod
        ref_label = "basket_weave" if "basket_weave" in cubes else list(cubes)[-1]
        fv_ref = eval_masks[ref_label][1]
        mom_ref = evaluation.moment0_map(ref_cubes[ref_label], dv_line & fv_ref, grid.channel_width_ghz,
                                         src.center_freq_ghz)
        maps = {k: evaluation.moment0_map(c, dv_line & fv_ref, grid.channel_width_ghz, src.center_freq_ghz)
                for k, c in cubes.items()}
        plotting.plot_scan_maps(outdir / "fig23_scan_maps.png", maps, mom_ref, map_grid.ra_arcsec,
                                map_grid.dec_arcsec, metrics)
        i_c = int(np.argmin(np.abs(map_grid.ra_arcsec - src.center_ra_arcsec)))
        j_c = int(np.argmin(np.abs(map_grid.dec_arcsec - src.center_dec_arcsec)))
        fwhm_ghz = src.center_freq_ghz * src.line_fwhm_kms / 299792.458
        plotting.plot_scan_center_spectrum(
            outdir / "fig24_scan_center_spectrum.png", grid.freq_ghz, fv_ref, truth_grid[i_c, j_c],
            ref_cubes[ref_label][i_c, j_c], {k: c[i_c, j_c] for k, c in cubes.items()},
            (src.center_freq_ghz - 4 * fwhm_ghz, src.center_freq_ghz + 4 * fwhm_ghz))
        for name, c in psd.items():
            plotting.plot_psd_vs_thermal(outdir / f"fig25_psd_{name}.png", c, title_prefix=f"{name}: ")
    return summary


def print_summary(summary):
    print(f"{'label':<20}{'RMSE vs ref [K]':>16}{'mom0 残差RMS':>14}{'フラックス比':>12}{'位置誤差 [\"]':>12}{'反復':>8}")
    for label, m in summary["metrics"].items():
        print(f"{label:<20}{m['cube_rmse_vs_reference_k']:>16.4f}{m['mom0_residual_rms_kkms']:>14.4f}"
              f"{m['flux_recovery_vs_reference']:>14.4f}{m['peak_position_error_arcsec']:>12.2f}"
              f"{str(m['n_iter']):>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="設定ファイル(既定: config/step2_scan.yaml)")
    parser.add_argument("--outdir", default=str(HERE / "outputs" / "step2_scan"))
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="設定の上書き。複数指定可")
    parser.add_argument("--joint-basket-weave", action="store_true",
                        help="basket_weave で x と y をまとめて反復する結果も出す(輝線モデルを共有)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, parse_set_args(args.set))
    summary = run(cfg, args.outdir, make_plots=not args.no_plots, joint=args.joint_basket_weave,
                  cli_argv=sys.argv, config_path=str(args.config))
    print_summary(summary)
    print(f"出力先: {args.outdir}")


if __name__ == "__main__":
    main()
