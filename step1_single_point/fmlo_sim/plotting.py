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


# ---------------------------------------------------------------------------
# PSD と理論熱雑音レベル(マイルストーン 1)
# ---------------------------------------------------------------------------

SERIES_COLORS = [INK_SECONDARY, BLUE, ORANGE, AQUA, "#7b5ea7", "#b8860b"]


def plot_psd_vs_thermal(path, cmp, title_prefix=""):
    """除去前後の PSD と、理論的な熱雑音(白色)レベル

    cmp : evaluation.psd_comparison の戻り値。灰色の帯が「低周波域」(low_freq_hz 未満)
    """
    freqs = cmp["freqs_hz"]
    titles = ["Mean of per-channel PSDs (line-free channels)",
              "PSD of the channel-averaged time series (common mode)"]
    whites = [cmp["white_per_channel"], cmp["white_common"]]
    keys = ["low_freq_per_channel", "low_freq_common"]
    f_low = next(iter(cmp["excess"].values()))["low_freq_hz"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, ax in enumerate(axs):
        for (label, psd), color in zip(cmp["psd"].items(), SERIES_COLORS):
            ratio = cmp["excess"][label][keys[i]]
            tag = f" (low-f / thermal = {ratio:.3g})" if ratio is not None else ""
            ax.loglog(freqs, np.maximum(psd[i], 1e-300), color=color, linewidth=1.1, label=f"{label}{tag}")
        ax.axhline(whites[i], color=INK, linestyle="--", linewidth=1.0, label="Expected thermal noise (white)")
        ax.axvspan(freqs[0], f_low, color=GRID, alpha=0.5, lw=0)
        ax.set_title(f"{title_prefix}{titles[i]}")
        ax.set_xlabel("Temporal frequency (Hz)")
        ax.set_ylabel("Power spectral density (K^2/Hz)")
        ax.grid(True, which="major")
        ax.legend(loc="best", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 輝線幅スイープ(マイルストーン 2)
# ---------------------------------------------------------------------------

def plot_sweep(path, rows, expected_thermal_rms_k=None):
    """輝線幅 / 変調全幅 vs 積分強度比・ピーク比・残差 RMS

    rows : run_step2_sweep の各条件の辞書(line_fwhm_over_mod_width, half_width_ch, method, ...)
    """
    methods = sorted({r["method"] for r in rows})
    widths = sorted({r["half_width_ch"] for r in rows})
    method_color = dict(zip(methods, [BLUE, ORANGE, AQUA, INK_SECONDARY]))
    markers = ["o", "s", "^", "D", "v", "P"]
    panels = [("integrated_ratio", "Integrated intensity ratio (restored / true)", 1.0),
              ("peak_ratio", "Peak ratio (restored / true)", 1.0),
              ("line_free_residual_rms_k", "Line-free residual RMS (K)", expected_thermal_rms_k)]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.0))
    for ax, (key, ylabel, ref) in zip(axs, panels):
        for m in methods:
            for w, mk in zip(widths, markers):
                sel = sorted([r for r in rows if r["method"] == m and r["half_width_ch"] == w],
                             key=lambda r: r["line_fwhm_over_mod_width"])
                if not sel:
                    continue
                x = [r["line_fwhm_over_mod_width"] for r in sel]
                y = [np.nan if r[key] is None else r[key] for r in sel]
                ax.plot(x, y, color=method_color[m], marker=mk, linewidth=1.2,
                        linestyle="-" if m == "iterative_pca" else "--", label=f"{m}, half width {w} ch")
        if ref is not None:
            label = "Expected thermal RMS" if key == "line_free_residual_rms_k" else None
            ax.axhline(ref, color=INK, linestyle=":" if label else "-", linewidth=0.9, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Line FWHM / full modulation width (dimensionless)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        ax.legend(loc="best", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# スキャン(マイルストーン 3)
# ---------------------------------------------------------------------------

def _map_extent(ra, dec):
    h_ra = (ra[1] - ra[0]) / 2.0 if ra.size > 1 else 0.5
    h_dec = (dec[1] - dec[0]) / 2.0 if dec.size > 1 else 0.5
    return [ra[0] - h_ra, ra[-1] + h_ra, dec[0] - h_dec, dec[-1] + h_dec]


def plot_scan_trajectory(path, passes, mom0_truth, ra_axis, dec_axis, beam_fwhm_arcsec=None):
    """スキャン軌跡(正解の積分強度マップに重ねる)と、時間 vs 指向位置"""
    n = len(passes)
    fig, axs = plt.subplots(1, n + 1, figsize=(6.0 * (n + 1), 5.0), squeeze=False)
    axs = axs[0]
    colors = [BLUE, ORANGE]
    for ax, p, c in zip(axs[:n], passes, colors):
        im = ax.imshow(mom0_truth.T, origin="lower", cmap=CMAP_SEQ, extent=_map_extent(ra_axis, dec_axis))
        ax.plot(p.ra_arcsec, p.dec_arcsec, color=c, linewidth=0.5, alpha=0.9)
        ax.plot(p.ra_arcsec[0], p.dec_arcsec[0], marker="o", color=INK, linestyle="none", label="Start")
        if beam_fwhm_arcsec:
            th = np.linspace(0, 2 * np.pi, 100)
            r = beam_fwhm_arcsec / 2.0
            x0, y0 = ra_axis[0] + 3 * r, dec_axis[0] + 3 * r
            ax.plot(x0 + r * np.cos(th), y0 + r * np.sin(th), color=INK, linewidth=1.0, label="Beam FWHM")
        ax.set_title(f"{p.name}: trajectory ({p.n_time} samples)")
        ax.set_xlabel("RA offset (arcsec)")
        ax.set_ylabel("Dec offset (arcsec)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, label="True integrated intensity (K km/s)")
    ax = axs[n]
    for p, c in zip(passes, colors):
        ax.plot(p.time_s, p.ra_arcsec, color=c, linewidth=0.8, label=f"{p.name}: RA")
        ax.plot(p.time_s, p.dec_arcsec, color=c, linewidth=0.8, linestyle=":", label=f"{p.name}: Dec")
    ax.set_title("Pointing vs time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Offset (arcsec)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_scan_timestreams(path, obs, cleaned, demod, freq_wide_ghz, max_rows=2000):
    """中間タイムストリーム:変調後の天体信号 / 観測データ / 除去後 / 復調後(空の周波数)

    時間方向は max_rows 行を超えないよう等間隔に間引いて表示する(平均はしない)。
    """
    p = obs.scan_pass
    step = max(1, int(np.ceil(p.n_time / max_rows)))
    t = p.time_s[::step]
    panels = [("Modulated source signal only", obs.signal_tod[::step], CMAP_SEQ, "seq"),
              ("Observed data (signal + noise)", obs.observed[::step], CMAP_DIV, "div"),
              ("After noise removal (X - C)", cleaned[::step], CMAP_DIV, "div")]
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.2))
    for ax, (title, tod, cmap, kind) in zip(axs[:3], panels):
        if kind == "seq":
            vmin, vmax = 0.0, float(np.max(tod)) or 1.0
        else:
            vmin, vmax = _signed_limits(tod)
        _tod_image(ax, tod, t, cmap, vmin, vmax, f"{p.name}: {title}", "Intensity (K)")
    d = demod[::step]
    vmin, vmax = _signed_limits(d[np.isfinite(d)])
    dt = t[1] - t[0] if t.size > 1 else 1.0
    im = axs[3].imshow(np.ma.masked_invalid(d), aspect="auto", origin="lower", cmap=CMAP_DIV, vmin=vmin,
                       vmax=vmax, interpolation="nearest",
                       extent=[freq_wide_ghz[0], freq_wide_ghz[-1], t[0], t[-1] + dt])
    axs[3].set_title(f"{p.name}: demodulated (sky frequency, blank = not observed)")
    axs[3].set_xlabel("Sky frequency (GHz)")
    axs[3].set_ylabel("Time (s)")
    fig.colorbar(im, ax=axs[3], label="Intensity (K)")
    _save(fig, path)


def plot_scan_maps(path, maps: dict, reference, ra_axis, dec_axis, metrics: dict = None):
    """積分強度マップの並列比較(上段)と、基準マップとの残差(下段)

    maps : {'x_scan': mom0, 'y_scan': mom0, 'basket_weave': mom0} [K km/s]
    reference : 雑音なしの天体信号を同じ手順でグリッディングした積分強度マップ
    """
    names = list(maps)
    n = len(names) + 1
    fig, axs = plt.subplots(2, n, figsize=(5.0 * n, 9.4), squeeze=False)
    ext = _map_extent(ra_axis, dec_axis)
    vmax = float(np.nanmax(reference)) or 1.0
    im = axs[0, 0].imshow(reference.T, origin="lower", cmap=CMAP_SEQ, vmin=0, vmax=vmax, extent=ext)
    axs[0, 0].set_title("Reference (noise-free signal, same gridding)")
    fig.colorbar(im, ax=axs[0, 0], label="Integrated intensity (K km/s)")
    resid = {k: maps[k] - reference for k in names}
    rmax = max([float(np.nanpercentile(np.abs(r), 99.5)) for r in resid.values()] + [1e-12])
    for i, k in enumerate(names, start=1):
        im = axs[0, i].imshow(maps[k].T, origin="lower", cmap=CMAP_SEQ, vmin=0, vmax=vmax, extent=ext)
        axs[0, i].set_title(f"Restored: {k}")
        fig.colorbar(im, ax=axs[0, i], label="Integrated intensity (K km/s)")
        im = axs[1, i].imshow(resid[k].T, origin="lower", cmap=CMAP_DIV, vmin=-rmax, vmax=rmax, extent=ext)
        tag = ""
        if metrics and k in metrics and metrics[k].get("mom0_residual_rms_kkms") is not None:
            tag = f"\nRMS {metrics[k]['mom0_residual_rms_kkms']:.3g} K km/s"
        axs[1, i].set_title(f"Residual: {k} - reference{tag}")
        fig.colorbar(im, ax=axs[1, i], label="Residual (K km/s)")
    axs[1, 0].axis("off")
    for ax in axs.ravel():
        if ax.axison:
            ax.set_xlabel("RA offset (arcsec)")
            ax.set_ylabel("Dec offset (arcsec)")
    _save(fig, path)


def plot_scan_center_spectrum(path, freq_ghz, valid, truth, reference, restored: dict, zoom_ghz=None):
    """天体中心の画素の復元スペクトルと正解スペクトル"""
    colors = [BLUE, ORANGE, AQUA, INK_SECONDARY]
    fig, axs = plt.subplots(2, 1, figsize=(11, 8))
    f = freq_ghz[valid]
    for ax in axs:
        for (k, spec), c in zip(restored.items(), colors):
            ax.plot(f, spec[valid], color=c, linewidth=0.9, label=f"Restored: {k}")
        ax.plot(f, reference[valid], color=INK_SECONDARY, linewidth=1.2, linestyle="--",
                label="Reference (noise-free signal, same gridding)")
        ax.plot(f, truth[valid], color=INK, linewidth=1.5, label="Ground truth (beam-convolved)")
        ax.set_xlabel("Sky frequency (GHz)")
        ax.set_ylabel("Intensity (K)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
    axs[0].set_title("Spectrum at the source-center pixel (frequencies observed in every sample)")
    axs[1].set_title("Zoom on the line")
    if zoom_ghz is not None:
        axs[1].set_xlim(*zoom_ghz)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 従来 OTF と FMLO の比較(Taniguchi et al. 2020 Fig. 14 の形式)
# ---------------------------------------------------------------------------

def _sky_axes(ax, ra_axis, dec_axis):
    """RA を東が左(論文と同じ向き)にする"""
    ext = _map_extent(ra_axis, dec_axis)
    ax.set_xlim(ext[1], ext[0])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Relative RA (arcsec)")
    ax.set_ylabel("Relative Dec (arcsec)")


def plot_otf_vs_fmlo(path, maps: dict, ra_axis, dec_axis, row_labels: dict, vmax=None):
    """積分強度マップを行 = スキャン(x / y / basket weave)、列 = OTF / FMLO で並べる

    maps : {('OTF', 'x_scan'): mom0, ('FMLO', 'x_scan'): mom0, ...} [K km/s]
    row_labels : {'x_scan': 'x-scan', ...}(パネル内の文字)。全パネル共通のカラーレンジ
    """
    rows = list(row_labels)
    cols = ["OTF", "FMLO"]
    if vmax is None:
        vmax = max(float(np.nanmax(m)) for m in maps.values())
    fig, axs = plt.subplots(len(rows), 2, figsize=(9.6, 4.2 * len(rows) + 0.8), squeeze=False)
    ext = _map_extent(ra_axis, dec_axis)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axs[i, j]
            im = ax.imshow(maps[(c, r)].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax, extent=ext,
                           interpolation="nearest")
            _sky_axes(ax, ra_axis, dec_axis)
            ax.text(0.04, 0.96, f"{c}\n{row_labels[r]}", transform=ax.transAxes, va="top", ha="left",
                    color="white", fontsize=12)
            if i == 0:
                ax.set_title(f"Integrated intensity ({c})")
    for j in range(2):
        fig.colorbar(im, ax=axs[:, j], orientation="horizontal", location="bottom", shrink=0.9, aspect=30,
                     label="Integrated intensity (K km/s)")
    _save(fig, path)


def plot_otf_vs_fmlo_residuals(path, maps: dict, references: dict, ra_axis, dec_axis, row_labels: dict,
                               stats: dict = None):
    """行ごとに 基準(雑音なし)/ OTF − 基準 / FMLO − 基準 の積分強度マップ"""
    rows = list(row_labels)
    fig, axs = plt.subplots(len(rows), 3, figsize=(15.0, 4.6 * len(rows)), squeeze=False)
    ext = _map_extent(ra_axis, dec_axis)
    vmax = max(float(np.nanmax(references[r])) for r in rows)
    resid = {k: maps[k] - references[k[1]] for k in maps}
    rmax = max(float(np.nanpercentile(np.abs(v), 99.5)) for v in resid.values()) or 1.0
    for i, r in enumerate(rows):
        im = axs[i, 0].imshow(references[r].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax, extent=ext)
        axs[i, 0].set_title(f"{row_labels[r]}: reference (noise-free signal)")
        fig.colorbar(im, ax=axs[i, 0], label="Integrated intensity (K km/s)")
        for j, c in enumerate(["OTF", "FMLO"], start=1):
            im = axs[i, j].imshow(resid[(c, r)].T, origin="lower", cmap=CMAP_DIV, vmin=-rmax, vmax=rmax,
                                  extent=ext)
            tag = ""
            if stats and (c, r) in stats:
                s = stats[(c, r)]
                tag = f"\nresidual RMS {s['mom0_residual_rms_kkms']:.3g} K km/s, flux {s['flux_ratio']:.3f}"
            axs[i, j].set_title(f"{row_labels[r]}: {c} − reference{tag}")
            fig.colorbar(im, ax=axs[i, j], label="Residual (K km/s)")
        for ax in axs[i]:
            _sky_axes(ax, ra_axis, dec_axis)
    _save(fig, path)
