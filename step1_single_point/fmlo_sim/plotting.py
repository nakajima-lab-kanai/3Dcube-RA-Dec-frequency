"""可視化(1関数1図)。軸ラベルには必ず単位を付ける"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .units import freq_width_to_velocity_width

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

# 系列の色は系列ごとに固定する(図をまたいで同じ色)
MODE_COLORS = {"none": BLUE, "channel_mean": ORANGE}
MODE_LABELS = {"none": "PCA, mean_mode = none", "channel_mean": "PCA, mean_mode = channel_mean"}

# 大きさ(0 以上)は青の単色、符号付きの量は青–灰–橙の発散型
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"]
)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_orange", ["#104281", BLUE, "#f0efec", ORANGE, "#8a3311"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "font.size": 10,
    "figure.constrained_layout.use": True,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tod_image(ax, tod, time_s, cmap, vmin, vmax, title, cbar_label):
    dt = time_s[1] - time_s[0]
    im = ax.imshow(
        tod, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[-0.5, tod.shape[1] - 0.5, time_s[0], time_s[-1] + dt],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Spectrometer channel (ch)")
    ax.set_ylabel("Time (s)")
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    return im


def _signed_limits(tod):
    """符号付きデータの表示範囲。中央値を中心に対称に取る"""
    center = float(np.median(tod))
    half = float(np.percentile(np.abs(tod - center), 99.5))
    half = half if half > 0 else 1.0
    return center - half, center + half


def plot_cube(path, sim):
    """キューブの積分強度マップと、取り出したピクセルのスペクトル"""
    grid, cfg = sim.grid, sim.cfg
    dv = freq_width_to_velocity_width(grid.channel_width_ghz, cfg.source.center_freq_ghz)
    mom0 = sim.cube.sum(axis=2) * dv
    half_pix = cfg.cube.pixel_arcsec / 2.0
    ra, dec = sim.ra_arcsec, sim.dec_arcsec

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    im = axs[0].imshow(
        mom0.T, origin="lower", cmap=CMAP_SEQ,
        extent=[ra[0] - half_pix, ra[-1] + half_pix, dec[0] - half_pix, dec[-1] + half_pix],
    )
    axs[0].plot(ra[sim.pixel[0]], dec[sim.pixel[1]], marker="+", ms=14, mew=2, color=ORANGE,
                linestyle="none")
    axs[0].set_title("Ground-truth cube: integrated intensity (+ = extracted pixel)")
    axs[0].set_xlabel("RA offset (arcsec)")
    axs[0].set_ylabel("Dec offset (arcsec)")
    fig.colorbar(im, ax=axs[0], label="Integrated intensity (K km/s)")

    f0 = grid.freq_ghz[grid.start0]
    f1 = grid.freq_ghz[grid.start0 + grid.n_channels - 1]
    axs[1].axvspan(f0, f1, color=GRID, alpha=0.6, lw=0, label="Spectrometer band at zero shift")
    axs[1].plot(grid.freq_ghz, sim.truth_k, color=BLUE, label="Spectrum at extracted pixel")
    axs[1].set_title("Ground-truth spectrum (cube frequency range is wider than the band)")
    axs[1].set_xlabel("Sky frequency (GHz)")
    axs[1].set_ylabel("Intensity (K)")
    axs[1].grid(True, axis="y")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _save(fig, path)


def plot_shift_pattern(path, sim):
    """変調パターン(時間 vs シフト量)"""
    dnu_mhz = sim.grid.channel_width_ghz * 1e3
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(sim.time_s, sim.shifts_ch, color=BLUE, drawstyle="steps-post", linewidth=1.2)
    ax.set_title(
        f"FMLO modulation pattern: {sim.cfg.modulation.pattern} "
        f"(1 ch = {dnu_mhz:.4f} MHz, positive = window moves to higher frequency)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Window shift (ch)")
    ax.grid(True)
    _save(fig, path)


def _line_channel_range(sim):
    """輝線が動く範囲のチャンネル(拡大表示用)"""
    grid, cfg = sim.grid, sim.cfg
    i_line = int(np.argmin(np.abs(grid.freq_ghz - cfg.source.center_freq_ghz))) - grid.start0
    fwhm_ch = cfg.source.center_freq_ghz * cfg.source.line_fwhm_kms / 299792.458 / grid.channel_width_ghz
    half = cfg.modulation.half_width_ch + 2.0 * fwhm_ch
    return max(i_line - half, 0), min(i_line + half, grid.n_channels - 1)


def plot_modulated_signal(path, sim):
    """変調後の天体信号のみの時間×チャンネル図"""
    vmax = float(sim.signal_tod.max()) or 1.0
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    _tod_image(axs[0], sim.signal_tod, sim.time_s, CMAP_SEQ, 0, vmax,
               "Modulated source signal only (full band)", "Intensity (K)")
    _tod_image(axs[1], sim.signal_tod, sim.time_s, CMAP_SEQ, 0, vmax,
               "Modulated source signal only (zoom on the line)", "Intensity (K)")
    axs[1].set_xlim(*_line_channel_range(sim))
    _save(fig, path)


def plot_noise(path, sim):
    """雑音の各成分と、観測データ全体の時間×チャンネル図"""
    panels = [(f"Noise component: {name}", tod) for name, tod in sim.noise.components.items()]
    panels.append(("Observed data (signal + all noise)", sim.observed))
    fig, axs = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)
    for ax, (title, tod) in zip(axs[0], panels):
        vmin, vmax = _signed_limits(tod)
        _tod_image(ax, tod, sim.time_s, CMAP_DIV, vmin, vmax, title, "Intensity (K)")
    _save(fig, path)


def plot_pca(path, sim, red):
    """PCA で除去した成分と、除去後のデータ"""
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
    vmin, vmax = _signed_limits(red.pca.removed)
    _tod_image(axs[0], red.pca.removed, sim.time_s, CMAP_DIV, vmin, vmax,
               f"Removed by PCA ({red.n_components} comp., mean_mode = {red.mean_mode})",
               "Intensity (K)")
    vmin, vmax = _signed_limits(red.pca.cleaned)
    _tod_image(axs[1], red.pca.cleaned, sim.time_s, CMAP_DIV, vmin, vmax,
               "Data after PCA removal", "Intensity (K)")

    ch = np.arange(sim.grid.n_channels)
    colors = [BLUE, ORANGE, AQUA]
    for i, comp in enumerate(red.pca.components[:3]):
        axs[2].plot(ch, comp, color=colors[i], linewidth=1.0,
                    label=f"Component {i + 1} ({100 * red.pca.explained_ratio[i]:.1f} % of variance)")
    axs[2].set_title("Spectra of the removed principal components")
    axs[2].set_xlabel("Spectrometer channel (ch)")
    axs[2].set_ylabel("Component amplitude (unit norm, dimensionless)")
    axs[2].grid(True)
    if len(red.pca.components):
        axs[2].legend(loc="best")
    _save(fig, path)


def plot_restored(path, sim, results):
    """復元スペクトルと正解スペクトルの重ね描き"""
    grid, cfg = sim.grid, sim.cfg
    freq = grid.freq_ghz
    valid = results["none"].hits == cfg.n_time
    fwhm_ghz = cfg.source.center_freq_ghz * cfg.source.line_fwhm_kms / 299792.458
    zoom = (cfg.source.center_freq_ghz - 4 * fwhm_ghz, cfg.source.center_freq_ghz + 4 * fwhm_ghz)

    fig, axs = plt.subplots(3, 1, figsize=(11, 10.5))
    for ax, title in zip(axs[:2], ["Restored vs ground-truth spectrum (fully covered range)",
                                   "Zoom on the line"]):
        for mode in ("none", "channel_mean"):
            ax.plot(freq[valid], results[mode].restored_k[valid], color=MODE_COLORS[mode],
                    linewidth=1.0, label=MODE_LABELS[mode])
        ax.plot(freq[valid], sim.truth_k[valid], color=INK, linewidth=1.6, label="Ground truth")
        ax.set_title(title)
        ax.set_ylabel("Intensity (K)")
        ax.grid(True)
        ax.legend(loc="upper right")
    axs[1].set_xlim(*zoom)

    for mode in ("none", "channel_mean"):
        axs[2].plot(freq[valid], (results[mode].restored_k - sim.truth_k)[valid],
                    color=MODE_COLORS[mode], linewidth=1.0, label=MODE_LABELS[mode])
    axs[2].axhline(0, color=INK, linewidth=0.8)
    axs[2].set_title("Residual (restored - ground truth), zoom on the line")
    axs[2].set_xlim(*zoom)
    axs[2].set_ylabel("Residual (K)")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")
    for ax in axs:
        ax.set_xlabel("Sky frequency (GHz)")
    _save(fig, path)


def plot_power_spectra(path, psd):
    """PCA 前後の雑音の時間方向パワースペクトル

    psd: {'before' | 'none' | 'channel_mean': (freqs_hz, psd_per_channel, psd_common)}
    """
    series = [("before", INK_SECONDARY, "Before PCA (observed data)"),
              ("none", MODE_COLORS["none"], "After " + MODE_LABELS["none"]),
              ("channel_mean", MODE_COLORS["channel_mean"], "After " + MODE_LABELS["channel_mean"])]
    titles = ["Mean of per-channel power spectra (line-free channels)",
              "Power spectrum of the channel-averaged time series (common mode)"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, ax in enumerate(axs):
        for key, color, label in series:
            freqs = psd[key][0]
            ax.loglog(freqs, np.maximum(psd[key][i + 1], 1e-300), color=color, label=label)
        ax.set_title(titles[i])
        ax.set_xlabel("Temporal frequency (Hz)")
        ax.set_ylabel("Power spectral density (K$^2$/Hz)")
        ax.grid(True, which="major")
        ax.legend(loc="best")
    _save(fig, path)


def plot_cube_3d(path, sim):
    """RA × Dec × 速度の3D等値面(HTML)。plotly が必要なので、呼ばれたときだけ import する"""
    from . import plotting3d
    plotting3d.plot_cube_3d(path, sim)
