"""キューブ(RA × Dec × 周波数)の3D可視化。plotly の等値面で、自己完結した HTML を1枚出力する

plotly は任意依存。通常の実行(--plot3d なし)では import されない。
保存済みの cube.npz から単独で描くこともできる:
    python -m fmlo_sim.plotting3d cube.npz --out cube3d.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .units import freq_to_velocity, velocity_width_to_freq_width


def require_plotly():
    """plotly を import して返す。未インストールなら日本語のエラーを出す"""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "3D 可視化には plotly が必要です。`pip install plotly` でインストールしてください"
            "(plotly は任意依存で、--plot3d を指定しない通常の実行には不要です)"
        ) from e
    return go


def plot_cube_3d_arrays(path, cube, ra_arcsec, dec_arcsec, freq_ghz, center_freq_ghz,
                        line_fwhm_kms, pointing_ra_arcsec, pointing_dec_arcsec, viz):
    """キューブを等値面で描いた HTML を path に保存する

    cube[i_ra, i_dec, i_freq] [K]。viz は config の VisualizationConfig。
    周波数方向は輝線中心 ±(viz.cube3d_half_range_n_fwhm × FWHM) に切り出す。
    等値面の値は、切り出した範囲のピーク値に対する割合 viz.cube3d_iso_fractions で指定する。
    """
    go = require_plotly()

    # 周波数方向を輝線中心の周りに切り出す(変調の余白を除く)
    half_ghz = viz.cube3d_half_range_n_fwhm * velocity_width_to_freq_width(line_fwhm_kms, center_freq_ghz)
    sel = np.abs(freq_ghz - center_freq_ghz) <= half_ghz
    if sel.sum() < 2:
        raise ValueError("切り出し範囲に周波数チャンネルが2つ未満しかありません。"
                         "visualization.cube3d_half_range_n_fwhm を大きくしてください")
    idx = np.flatnonzero(sel)
    # HTML のサイズを抑えるため、点数が上限を超えるときは等間隔に間引く
    stride = max(1, int(np.ceil(idx.size / viz.cube3d_max_freq_points)))
    idx = idx[::stride]
    # 速度軸は昇順にそろえる(周波数が高いほど速度は小さい)
    idx = idx[np.argsort(freq_to_velocity(freq_ghz[idx], center_freq_ghz))]
    sub = np.asarray(cube[:, :, idx], dtype=np.float64)
    freq = freq_ghz[idx]
    vel = freq_to_velocity(freq, center_freq_ghz)

    peak = float(sub.max())
    if peak <= 0:
        raise ValueError("切り出した範囲のピーク値が 0 以下で、等値面を描けません")

    ra_g, dec_g, vel_g = np.meshgrid(ra_arcsec, dec_arcsec, vel, indexing="ij")
    freq_g = np.broadcast_to(freq, sub.shape)
    # 等値面のトレースごとに全データが HTML に入るので、float32 にしてサイズを抑える
    flat = {k: np.ascontiguousarray(v.ravel(), dtype=np.float32) for k, v in
            dict(x=ra_g, y=dec_g, z=vel_g, value=sub, customdata=freq_g).items()}
    hover = ("RA: %{x:.0f} arcsec<br>Dec: %{y:.0f} arcsec<br>速度: %{z:.1f} km/s"
             "<br>周波数: %{customdata:.4f} GHz<br>強度: %{value:.3f} K<extra></extra>")

    fig = go.Figure()
    for frac in sorted(viz.cube3d_iso_fractions):
        level = frac * peak
        fig.add_trace(go.Isosurface(
            **flat, isomin=level, isomax=level, surface_count=1, caps=dict(x_show=False, y_show=False,
                                                                           z_show=False),
            opacity=viz.cube3d_opacity, coloraxis="coloraxis", hovertemplate=hover,
            name=f"{frac:g} × ピーク({level:.3g} K)", showlegend=True,
        ))

    # 観測した点の位置を、速度方向の細い線で示す(キューブのピクセルに丸めた位置)
    p_ra = float(ra_arcsec[np.argmin(np.abs(ra_arcsec - pointing_ra_arcsec))])
    p_dec = float(dec_arcsec[np.argmin(np.abs(dec_arcsec - pointing_dec_arcsec))])
    fig.add_trace(go.Scatter3d(
        x=[p_ra, p_ra], y=[p_dec, p_dec], z=[float(vel.min()), float(vel.max())],
        mode="lines", line=dict(color="#0b0b0b", width=3, dash="dash"),
        name=f"観測点(RA {p_ra:g}, Dec {p_dec:g} arcsec)", hoverinfo="name",
    ))

    fig.update_layout(
        title=f"キューブの等値面(中心 {center_freq_ghz:g} GHz、ピーク {peak:.3g} K)",
        scene=dict(xaxis_title="RA オフセット [arcsec]", yaxis_title="Dec オフセット [arcsec]",
                   zaxis_title="視線速度 [km/s]", aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=1)),
        coloraxis=dict(colorscale="Viridis", cmin=0.0, cmax=peak,
                       colorbar=dict(title="強度 [K]")),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    # plotly.js を HTML に埋め込み、オフラインでも開けるようにする
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)


def plot_cube_3d(path, sim):
    """Simulation の正解キューブを3Dで描く(plot_cube_3d_arrays のラッパー)"""
    cfg = sim.cfg
    i_ra, i_dec = sim.pixel
    plot_cube_3d_arrays(
        path, sim.cube, sim.ra_arcsec, sim.dec_arcsec, sim.grid.freq_ghz,
        cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        float(sim.ra_arcsec[i_ra]), float(sim.dec_arcsec[i_dec]), cfg.visualization,
    )


def main():
    from .cube import load_cube
    from .config import load_config, parse_set_args

    parser = argparse.ArgumentParser(description="保存済みの cube.npz を3D表示の HTML にする")
    parser.add_argument("cube", help="cube.npz のパス")
    parser.add_argument("--out", default="cube3d.html", help="出力する HTML(既定: cube3d.html)")
    parser.add_argument("--config", default=None,
                        help="visualization の設定を読む YAML(既定: config/default.yaml)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="設定の上書き(例: --set visualization.cube3d_opacity=0.2)")
    args = parser.parse_args()

    require_plotly()
    cube, ra, dec, freq, meta = load_cube(args.cube)
    # 輝線・観測点の情報は npz に保存された設定から、描画の設定は現在の YAML から読む
    saved = meta["config"]
    src, pt = saved["source"], saved["pointing"]
    p_ra = src["center_ra_arcsec"] if pt["ra_arcsec"] is None else pt["ra_arcsec"]
    p_dec = src["center_dec_arcsec"] if pt["dec_arcsec"] is None else pt["dec_arcsec"]
    viz = load_config(args.config, parse_set_args(args.set)).visualization
    plot_cube_3d_arrays(args.out, cube, ra, dec, freq, src["center_freq_ghz"],
                        src["line_fwhm_kms"], p_ra, p_dec, viz)
    print(f"出力: {Path(args.out)}")


if __name__ == "__main__":
    main()
