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


def plot_fig6_diagnostics(path, result, time_s, ch_range=None, title_suffix=""):
    """Taniguchi et al. (2020) Fig. 6 に対応する 2行×4列の成分推定診断図

    行構成:
      Row 1 (1st estimate): X, C_1, A_1, E_1
      Row 2 (2nd estimate): U_2 (= X - A_1), C_2, A_2, E_2

    列ごとの表示規則:
      - Col 0 (Input TOD): X と U_2 で共通のカラーレンジ
      - Col 1 (Correlated TOD): C_1 と C_2 で共通のカラーレンジ
      - Col 2 (Estimated Sky Signal): A_1 と A_2 で共通のカラーレンジ
      - Col 3 (Residual TOD): E_1 と E_2 で共通の対称カラーレンジ (0 を中央とする)

    横軸は分光計チャンネル (IF 座標)。
    ch_range を指定した場合は描画のみを指定スライスに限定する (元データは不変)。
    """
    x = result.x_tod
    u1, c1, a1, e1 = result.iter1.u_tod, result.iter1.c_tod, result.iter1.a_tod, result.iter1.e_tod
    u2, c2, a2, e2 = result.iter2.u_tod, result.iter2.c_tod, result.iter2.a_tod, result.iter2.e_tod

    n_time, n_channels = x.shape
    dt = time_s[1] - time_s[0]

    if ch_range is not None:
        ch_start, ch_end = ch_range
        ch_start = max(0, int(ch_start))
        ch_end = min(n_channels, int(ch_end))
        ch_slice = slice(ch_start, ch_end)
        extent = [ch_start - 0.5, ch_end - 0.5, time_s[0], time_s[-1] + dt]
    else:
        ch_slice = slice(None)
        extent = [-0.5, n_channels - 0.5, time_s[0], time_s[-1] + dt]

    # スライス適用 (描画用)
    x_p = x[:, ch_slice]
    u1_p, c1_p, a1_p, e1_p = u1[:, ch_slice], c1[:, ch_slice], a1[:, ch_slice], e1[:, ch_slice]
    u2_p, c2_p, a2_p, e2_p = u2[:, ch_slice], c2[:, ch_slice], a2[:, ch_slice], e2[:, ch_slice]

    # 各列のカラーレンジ決定 (1回目と2回目で同一)
    # Col 0: Input (X, U_2)
    inputs = np.concatenate([x_p.ravel(), u2_p.ravel()])
    v0_min, v0_max = float(np.percentile(inputs, 0.5)), float(np.percentile(inputs, 99.5))

    # Col 1: Correlated (C_1, C_2)
    corrs = np.concatenate([c1_p.ravel(), c2_p.ravel()])
    v1_min, v1_max = float(np.percentile(corrs, 0.5)), float(np.percentile(corrs, 99.5))

    # Col 2: Signal (A_1, A_2)
    sigs = np.concatenate([a1_p.ravel(), a2_p.ravel()])
    v2_min = 0.0
    v2_max = max(float(np.percentile(sigs, 99.8)), 0.1)

    # Col 3: Residual (E_1, E_2) (符号付き・0 中心)
    resids = np.concatenate([e1_p.ravel(), e2_p.ravel()])
    e_abs_max = max(float(np.percentile(np.abs(resids), 99.5)), 0.1)
    v3_min, v3_max = -e_abs_max, e_abs_max

    # B-3: layout engine を "none" に設定して subplots_adjust を有効化し、
    # 全体タイトルと上段パネルタイトルの重なりを防ぐ。
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey=True, layout="none")
    fig.subplots_adjust(top=0.88)

    grid_data = [
        # Row 1: 1st estimate
        [
            (x_p, CMAP_SEQ, v0_min, v0_max, "$X$ (Observed TOD)", "Input: $X$ [K]"),
            (c1_p, CMAP_DIV, v1_min, v1_max, r"$\hat{T}_{\mathrm{cor},1}$ ($C_1$)", "Correlated [K]"),
            (a1_p, CMAP_SEQ, v2_min, v2_max, r"$\hat{T}_{\mathrm{ast},1}$ ($A_1$)", "Sky signal [K]"),
            (e1_p, CMAP_DIV, v3_min, v3_max, r"$\hat{E}_{\mathrm{nc},1}$ ($E_1 = X - C_1 - A_1$)", "Residual [K]"),
        ],
        # Row 2: 2nd estimate
        [
            (u2_p, CMAP_SEQ, v0_min, v0_max, r"$U_2 = X - \hat{T}_{\mathrm{ast},1}$", "Input: $X - A_1$ [K]"),
            (c2_p, CMAP_DIV, v1_min, v1_max, r"$\hat{T}_{\mathrm{cor},2}$ ($C_2$)", "Correlated [K]"),
            (a2_p, CMAP_SEQ, v2_min, v2_max, r"$\hat{T}_{\mathrm{ast},2}$ ($A_2$)", "Sky signal [K]"),
            (e2_p, CMAP_DIV, v3_min, v3_max, r"$\hat{E}_{\mathrm{nc},2}$ ($E_2 = X - C_2 - A_2$)", "Residual [K]"),
        ],
    ]

    row_titles = ["1st estimate (r=1)", "2nd estimate (r=2)"]

    for row_idx in range(2):
        for col_idx in range(4):
            ax = axs[row_idx, col_idx]
            data, cmap, vmin, vmax, title, cbar_label = grid_data[row_idx][col_idx]
            im = ax.imshow(
                data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=extent, interpolation="nearest",
            )
            ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{row_titles[row_idx]}\nTime (s)")
            else:
                ax.set_ylabel("")
            if row_idx == 1:
                ax.set_xlabel("Spectrometer channel (IF ch)")
            ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Taniguchi et al. (2020) Fig. 6 Component Diagnostics{title_suffix}\n"
        r"Identity: $X = C_r + A_r + E_r$ holds strictly for both $r=1$ and $r=2$.",
        fontsize=12,
        y=0.98,
    )
    _save(fig, path)


def plot_fig6_restored_spectra(
    path,
    freq_ghz,
    truth_k,
    restored1_k,
    restored2_k,
    hits,
    n_time,
    center_freq_ghz,
    line_fwhm_kms,
):
    """Fig. 6 の 1回目・2回目の復元スペクトルと正解スペクトルの重ね描き

    2×2 レイアウト:
      上段: 全範囲 (左) と輝線付近の拡大 (右) — 正解・1回目・2回目の重ね描き
      下段: 同じ範囲で 復元 − 正解 の残差

    評価時のみ正解を参照する。hits == n_time の領域のみ描画する。

    Parameters
    ----------
    path : 出力先パス
    freq_ghz : (n_wide,) sky 周波数 [GHz]
    truth_k : (n_wide,) 正解スペクトル [K]
    restored1_k : (n_wide,) 1回目の復元スペクトル [K]
    restored2_k : (n_wide,) 2回目の復元スペクトル [K]
    hits : (n_wide,) 観測ヒット数
    n_time : 時間サンプル数
    center_freq_ghz : 輝線中心周波数 [GHz]
    line_fwhm_kms : 輝線幅 FWHM [km/s]
    """
    valid = hits == n_time
    freq_v = freq_ghz[valid]
    truth_v = truth_k[valid]
    r1_v = restored1_k[valid]
    r2_v = restored2_k[valid]
    res1_v = r1_v - truth_v
    res2_v = r2_v - truth_v

    fwhm_ghz = center_freq_ghz * line_fwhm_kms / 299792.458
    zoom = (center_freq_ghz - 4 * fwhm_ghz, center_freq_ghz + 4 * fwhm_ghz)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # 上段: 正解・1回目・2回目の重ね描き
    for col, (ax, xlim, title) in enumerate(zip(
        axs[0],
        [None, zoom],
        ["Restored vs ground-truth spectrum (fully covered range)",
         "Zoom on the line"],
    )):
        ax.plot(freq_v, truth_v, color=INK, linewidth=1.6, label="Ground truth", zorder=3)
        ax.plot(freq_v, r1_v, color=BLUE, linewidth=1.0, label="1st estimate", alpha=0.85)
        ax.plot(freq_v, r2_v, color=ORANGE, linewidth=1.0, label="2nd estimate", alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Intensity (K)")
        ax.set_xlabel("Sky frequency (GHz)")
        ax.grid(True)
        ax.legend(loc="upper right")
        if xlim is not None:
            ax.set_xlim(*xlim)

    # 下段: 残差
    for col, (ax, xlim, title) in enumerate(zip(
        axs[1],
        [None, zoom],
        ["Residual (restored − ground truth), full range",
         "Residual (restored − ground truth), zoom on the line"],
    )):
        ax.plot(freq_v, res1_v, color=BLUE, linewidth=1.0, label="1st estimate", alpha=0.85)
        ax.plot(freq_v, res2_v, color=ORANGE, linewidth=1.0, label="2nd estimate", alpha=0.85)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Residual (K)")
        ax.set_xlabel("Sky frequency (GHz)")
        ax.grid(True)
        ax.legend(loc="upper right")
        if xlim is not None:
            ax.set_xlim(*xlim)

    fig.suptitle(
        "Taniguchi et al. (2020) Fig. 6: Restored spectra comparison\n"
        "(hits == n_time region only, ground truth used for evaluation only)",
        fontsize=12,
    )
    _save(fig, path)
