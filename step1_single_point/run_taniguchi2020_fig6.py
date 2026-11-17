"""Taniguchi et al. (2020) Fig. 6 の再現図を出力する

    python run_taniguchi2020_fig6.py [--config config/taniguchi2020_fig6.yaml]
        [--outdir outputs/taniguchi2020_fig6] [--set reduction.n_components=4,4]

出力: 再現図 (PNG)、各 timestream (NPZ)、設定と指標 (JSON)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fmlo_sim.taniguchi2020 import load_fig6_config, run_fig6, to_panel_pixels

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(HERE / "config" / "taniguchi2020_fig6.yaml"))
    p.add_argument("--outdir", default=str(HERE / "outputs" / "taniguchi2020_fig6"))
    p.add_argument("--set", dest="overrides", action="append", default=[])
    a = p.parse_args()
    ov = dict(s.split("=", 1) for s in a.overrides)
    cfg = load_fig6_config(a.config, ov)
    truth, res = run_fig6(cfg)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    pc = cfg.plot
    fig = plt.figure(figsize=(10, 5))
    xs = [95, 335, 575, 815]; ys = [500 - 13 - 160, 500 - 238 - 160]
    labels = [[r"$T^{\rm cal}$", r"$\hat{T}^{\rm cor}$", r"$\hat{T}^{\rm ast}$", r"$\hat{E}^{\rm nc}$"],
              [r"$T^{\rm cal}-\hat{T}^{\rm ast}$", r"$\hat{T}^{\rm cor}$", r"$\hat{T}^{\rm ast}$", r"$\hat{E}^{\rm nc}$"]]
    
    for row, st in enumerate(res.steps[:2]):
        for col, arr in enumerate([st.x_in, st.t_cor, st.t_ast, st.e_nc]):
            ax = fig.add_axes([xs[col] / 1000, ys[row] / 500, 160 / 1000, 160 / 500])
            ax.imshow(to_panel_pixels(arr, pc.panel_px), origin="lower", aspect="auto",
                      cmap=pc.cmap, vmin=pc.vmin, vmax=pc.vmax, interpolation="nearest")
            ax.set_axis_off()
            ax.text(0.03, 0.97, labels[row][col], transform=ax.transAxes, va="top", color="w", fontsize=16)
            if col < 3:
                fig.patches.append(matplotlib.patches.FancyArrow(
                    (xs[col] + 162) / 1000, (ys[row] + 80) / 500, 72 / 1000, 0, width=0.006,
                    head_width=0.03, head_length=0.012, color="k", transform=fig.transFigure))
        fig.text(0.04, (ys[row] + 80) / 500, ["1st iteration", "2nd iteration"][row],
                 rotation=90, va="center", ha="center", fontsize=14)
    pink, wine = "#ff9a8f", "#8e1650"
    L = matplotlib.lines.Line2D
    for row in range(2):
        yb, ym = ys[row] / 500, (ys[row] - 32) / 500     # パネル下端 / 引き算記号の高さ
        xa, xc = (xs[2] + 80) / 1000, (xs[0] + 80) / 1000
        fig.add_artist(L([xa, xa, xc + 0.035], [yb, ym, ym], color=pink, lw=4,
                         solid_joinstyle="miter", transform=fig.transFigure))
        fig.patches.append(matplotlib.patches.Polygon(
            [[xc + 0.018, ym], [xc + 0.038, ym + 0.022], [xc + 0.038, ym - 0.022]],
            color=pink, transform=fig.transFigure))
        fig.add_artist(L([xc, xc], [yb, ym], color=pink, lw=4, transform=fig.transFigure))
        if row == 0:   # 次の反復の入力パネル上端まで
            fig.add_artist(L([xc, xc], [ym, (ys[1] + 160) / 500], color=pink, lw=4, transform=fig.transFigure))
        else:          # 3 回目以降へ続く
            fig.add_artist(L([xc, xc], [ym - 0.035, 0.05], color=pink, lw=4, ls=":", transform=fig.transFigure))
        fig.patches.append(matplotlib.patches.Ellipse((xc, ym), 0.03, 0.06, fc=wine, ec=pink, lw=3,
                                                      transform=fig.transFigure, zorder=5))
        fig.text(xc, ym, "−", color="w", ha="center", va="center", fontsize=16, weight="bold", zorder=6)
        fig.text(xc - 0.025, ym, "subtraction", ha="right", va="center", fontsize=12)
    fig.text(0.04, 0.07, "...", rotation=90, ha="center", va="center", fontsize=14)
    fig.savefig(out / "fig6_taniguchi2020_reproduced.png", dpi=200)
    np.savez_compressed(out / "fig6_timestreams.npz", xi_bar=truth.xi_bar, t_cal=truth.t_cal,
                        **{f"{k}{i+1}": getattr(s, k) for i, s in enumerate(res.steps)
                           for k in ("x_in", "t_cor", "t_ast", "e_nc", "spectrum", "model")})
    
    lf = np.r_[0:500, 1700:cfg.spectrometer.n_channels]
    metrics = {
        "config": cfg.to_dict(),
        "d_tilde": int(res.hits.size),
        "fm_width_ch": int(truth.xi_bar.max()),
        "iterations": [
            {"n_components": s.n_components, "model_peak": float(s.model.max()),
             "model_n_channels": int(np.count_nonzero(s.model)),
             "residual_std_all": float(s.e_nc.std()),
             "residual_std_line_free": float(s.e_nc[:, lf].std()),
             "residual_min_time_mean_demod": float(np.nanmin(s.spectrum - s.model)),
             "variation": s.variation}
            for s in res.steps],
        "converged": res.converged,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    for i, m in enumerate(metrics["iterations"], 1):
        print(f"iter {i}: K={m['n_components']} model peak={m['model_peak']:.2f} "
              f"resid std={m['residual_std_all']:.3f} variation={m['variation']}")
    print("saved:", out / "fig6_taniguchi2020_reproduced.png")


if __name__ == "__main__":
    main()
