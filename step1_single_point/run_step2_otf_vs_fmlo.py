"""段階 2-B:従来の OTF と FMLO の積分強度マップの比較(Taniguchi et al. 2020 Fig. 14 の形式)

観測対象・スキャン軌跡・雑音モデル・FMLO の観測パラメータ・FMLO の処理は run_step2_scan.py と同じ
(config/step2_scan.yaml)。同じ軌跡で、LO を固定し OFF 点をはさむ従来の OTF 観測も模擬し
(fmlo_sim/otf.py)、x-scan・y-scan・basket weave の積分強度マップを OTF / FMLO で並べる。

積分範囲は論文と同じ考え方にする(comparison セクション):
  単一スキャン : 輝線と輝線のない速度を含む広い範囲
  basket weave : 輝線に絞った狭い範囲(論文は 50.5 km/s と 30.5 km/s。狭い方は広い方の約 0.6 倍)

使い方:
    python run_step2_otf_vs_fmlo.py
    python run_step2_otf_vs_fmlo.py --set otf.baseline_order=-1 --outdir outputs/step2_otf_vs_fmlo_nobaseline
"""
import argparse
import dataclasses
import csv
import json
import sys
from pathlib import Path

import numpy as np

from fmlo_sim import cube as cube_mod
from fmlo_sim import evaluation, mapping, otf, pipeline, plotting, scan
from fmlo_sim.config import load_config, parse_set_args
from fmlo_sim.provenance import collect_provenance
from fmlo_sim.units import freq_to_velocity
from run_fig6 import _get_git_info
from run_step2_scan import full_coverage_freq, scanned_pixels

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "step2_scan.yaml"
ROW_LABELS = {"x_scan": "x-scan", "y_scan": "y-scan", "basket_weave": "basket\nweave"}
CSV_KEYS = ["method", "scan", "vrange_kms", "mom0_peak_kkms", "mom0_peak_reference_kkms", "flux_ratio",
            "mom0_residual_rms_kkms", "mom0_offsource_rms_kkms", "cube_rmse_vs_reference_k",
            "cube_line_free_rms_k", "on_time_s", "total_time_s"]


def combine(method, a: mapping.GriddedCube, b: mapping.GriddedCube):
    if method == "weighted_mean":
        return mapping.combine_weighted_mean([a, b]).data
    return mapping.combine_fourier(a.data, b.data)


def map_stats(mom, mom_ref, spatial):
    """積分強度マップの評価(基準 = 雑音なしの天体信号を同じ手順で処理したもの)"""
    ok = spatial & np.isfinite(mom)
    peak_ref = float(np.nanmax(np.where(ok, mom_ref, np.nan)))
    off = ok & (mom_ref < 0.01 * peak_ref)        # 天体のない画素(基準の 1 % 未満)
    return {
        "mom0_peak_kkms": float(np.nanmax(np.where(ok, mom, np.nan))),
        "mom0_peak_reference_kkms": peak_ref,
        "flux_ratio": float(mom[ok].sum() / mom_ref[ok].sum()),
        "mom0_residual_rms_kkms": float(np.sqrt(np.mean((mom - mom_ref)[ok] ** 2))),
        "mom0_offsource_rms_kkms": float(np.sqrt(np.mean(mom[off] ** 2))) if off.any() else None,
        "n_offsource_pixels": int(off.sum()),
    }


def run(cfg, outdir, make_plots=True, cli_argv=None, config_path=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if cfg.scan.mode != "basket_weave":
        raise SystemExit("この比較は scan.mode = basket_weave で実行します")
    src, m = cfg.source, cfg.mapping
    comparison = dataclasses.asdict(cfg.comparison)

    sky = pipeline.make_internal_sky_cube(cfg)
    passes = scan.make_passes(cfg)
    map_grid = mapping.make_map_grid(cfg)
    wmats = {p.name: mapping.gridding_matrix(map_grid, p.ra_arcsec, p.dec_arcsec, m.kernel_fwhm_arcsec,
                                             m.kernel_support_fwhm) for p in passes}

    # FMLO(run_step2_scan.py と同じ観測・処理)
    f_obs = [scan.observe_pass(sky, cfg, p, i) for i, p in enumerate(passes)]
    grid = f_obs[0].grid
    start0, n_wide, n_ch = grid.start0, grid.n_wide, grid.n_channels
    f_red, f_ref = {}, {}
    for o in f_obs:
        name = o.scan_pass.name
        g = mapping.PassGeometry(name, o.scan_pass.ra_arcsec, o.scan_pass.dec_arcsec, o.shifts_ch, wmats[name])
        f_red[name] = mapping.reduce_passes(name, [o.observed], [g], map_grid, start0, n_wide, cfg.removal).gridded
        f_ref[name] = mapping.grid_tod(o.signal_tod, o.shifts_ch, start0, n_wide, wmats[name], map_grid.shape)

    # 従来の OTF(同じ軌跡・LO 固定・OFF 点を引く)
    in_line, line_free = evaluation.line_masks(grid.freq_ghz, src.center_freq_ghz, src.line_fwhm_kms,
                                               cfg.evaluation.line_free_n_fwhm)
    zeros = {p.name: np.zeros(p.n_time, dtype=int) for p in passes}
    free_ch = evaluation.line_free_channels(line_free, zeros[passes[0].name], start0, n_ch)
    o_obs = [otf.observe_pass(sky, cfg, p, i) for i, p in enumerate(passes)]
    o_red, o_ref, o_time = {}, {}, {}
    for o in o_obs:
        name = o.scan_pass.name
        r = otf.reduce_pass(o, free_ch, cfg.otf.off_interpolation, cfg.otf.baseline_order)
        o_red[name] = mapping.grid_tod(r.calibrated, zeros[name], start0, n_wide, wmats[name], map_grid.shape)
        o_ref[name] = mapping.grid_tod(o.signal_on, zeros[name], start0, n_wide, wmats[name], map_grid.shape)
        o_time[name] = otf.time_summary(o, cfg.modulation.dt_s)

    cubes = {("FMLO", n): c.data for n, c in f_red.items()}
    cubes.update({("OTF", n): c.data for n, c in o_red.items()})
    refs = {("FMLO", n): c.data for n, c in f_ref.items()}
    refs.update({("OTF", n): c.data for n, c in o_ref.items()})
    for meth, red, ref in (("FMLO", f_red, f_ref), ("OTF", o_red, o_ref)):
        cubes[(meth, "basket_weave")] = combine(m.basket_weave_method, red["x_scan"], red["y_scan"])
        refs[(meth, "basket_weave")] = combine(m.basket_weave_method, ref["x_scan"], ref["y_scan"])

    # 積分範囲(輝線中心からの速度。radio 定義)
    vel = freq_to_velocity(grid.freq_ghz, src.center_freq_ghz)
    half = {"x_scan": comparison["single_half_width_fwhm"] * src.line_fwhm_kms,
            "y_scan": comparison["single_half_width_fwhm"] * src.line_fwhm_kms,
            "basket_weave": comparison["basket_half_width_fwhm"] * src.line_fwhm_kms}
    f_valid = full_coverage_freq([o.shifts_ch for o in f_obs], start0, n_ch, n_wide)
    o_valid = np.zeros(n_wide, dtype=bool)
    o_valid[start0:start0 + n_ch] = True
    valid = {"FMLO": f_valid, "OTF": o_valid}
    vmask = {}
    for r, h in half.items():
        vmask[r] = np.abs(vel) <= h
        for meth, v in valid.items():
            if not np.all(v[vmask[r]]):
                raise SystemExit(f"{meth} の観測帯域が積分範囲 ±{h:.1f} km/s を覆っていません")
    dv = grid.channel_width_ghz
    mom = {k: evaluation.moment0_map(c, vmask[k[1]], dv, src.center_freq_ghz) for k, c in cubes.items()}
    mom_ref = {k: evaluation.moment0_map(c, vmask[k[1]], dv, src.center_freq_ghz) for k, c in refs.items()}

    # 評価
    rr, dd = map_grid.pixel_centers()
    truth_grid = cube_mod.sample_spectra(sky.data, sky.ra_arcsec, sky.dec_arcsec, rr, dd).reshape(
        *map_grid.shape, n_wide)
    pass_by_name = {p.name: p for p in passes}
    stats, rows = {}, []
    for k in cubes:
        meth, r = k
        ps = passes if r == "basket_weave" else [pass_by_name[r]]
        spatial = scanned_pixels(map_grid, ps)
        s = map_stats(mom[k], mom_ref[k], spatial)
        cube_eval = evaluation.evaluate_scan_cube(
            cubes[k], refs[k], truth_grid, spatial, valid[meth], in_line, line_free, map_grid.ra_arcsec,
            map_grid.dec_arcsec, dv, src.center_freq_ghz, map_grid.pixel_arcsec,
            src.center_ra_arcsec, src.center_dec_arcsec)
        if meth == "OTF":
            on_t = sum(o_time[p.name]["on_time_s"] for p in ps)
            tot_t = sum(o_time[p.name]["total_time_s"] for p in ps)
        else:
            on_t = tot_t = sum(p.n_time * cfg.modulation.dt_s for p in ps)
        s.update({"vrange_kms": [-half[r], half[r]], "cube_rmse_vs_reference_k": cube_eval["cube_rmse_vs_reference_k"],
                  "cube_line_free_rms_k": cube_eval["cube_line_free_rms_k"], "on_time_s": on_t, "total_time_s": tot_t})
        stats[k] = s
        rows.append({"method": meth, "scan": r, **s})

    summary = {
        "config": cfg.to_dict(),
        "sampling": scan.sampling_summary(cfg, passes),
        "otf_time": o_time,
        "metrics": rows,
        "provenance": collect_provenance(_get_git_info(), cli_argv, config_path),
    }
    with open(outdir / "metrics_otf_vs_fmlo.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    with open(outdir / "metrics_otf_vs_fmlo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_KEYS)
        for row in rows:
            w.writerow([json.dumps(row[c]) if isinstance(row[c], list) else row[c] for c in CSV_KEYS])
    np.savez_compressed(outdir / "mom0_otf_vs_fmlo.npz", ra_arcsec=map_grid.ra_arcsec,
                        dec_arcsec=map_grid.dec_arcsec,
                        **{f"{a}_{b}": v for (a, b), v in mom.items()},
                        **{f"reference_{a}_{b}": v for (a, b), v in mom_ref.items()})

    if make_plots:
        labels = {r: f"{ROW_LABELS[r]}" for r in ROW_LABELS}
        plotting.plot_otf_vs_fmlo(outdir / "fig31_otf_vs_fmlo_maps.png", mom, map_grid.ra_arcsec,
                                  map_grid.dec_arcsec, labels)
        plotting.plot_otf_vs_fmlo_residuals(
            outdir / "fig32_otf_vs_fmlo_residuals.png", mom, {r: mom_ref[("FMLO", r)] for r in ROW_LABELS},
            map_grid.ra_arcsec, map_grid.dec_arcsec, {r: ROW_LABELS[r].replace("\n", " ") for r in ROW_LABELS},
            stats)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="設定ファイル(既定: config/step2_scan.yaml)")
    parser.add_argument("--outdir", default=str(HERE / "outputs" / "step2_otf_vs_fmlo"))
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="設定の上書き。複数指定可")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, parse_set_args(args.set))
    summary = run(cfg, args.outdir, make_plots=not args.no_plots, cli_argv=sys.argv,
                  config_path=str(args.config))
    print(f"{'method':<6}{'scan':<14}{'積分範囲 [km/s]':>18}{'ピーク比':>10}{'フラックス比':>12}"
          f"{'残差RMS':>10}{'天体外RMS':>11}{'観測時間 [s]':>14}")
    for r in summary["metrics"]:
        v = r["vrange_kms"]
        print(f"{r['method']:<6}{r['scan']:<14}{f'{v[0]:.0f}〜{v[1]:+.0f}':>18}"
              f"{r['mom0_peak_kkms'] / r['mom0_peak_reference_kkms']:>10.3f}{r['flux_ratio']:>12.3f}"
              f"{r['mom0_residual_rms_kkms']:>10.3f}{r['mom0_offsource_rms_kkms']:>11.3f}{r['total_time_s']:>14.0f}")
    print(f"出力先: {args.outdir}")


if __name__ == "__main__":
    main()
