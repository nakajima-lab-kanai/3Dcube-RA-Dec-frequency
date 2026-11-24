"""復元スペクトルの評価"""
from __future__ import annotations

import numpy as np

from .units import freq_width_to_velocity_width, velocity_width_to_freq_width


def line_masks(freq_ghz, center_freq_ghz, line_fwhm_kms, n_fwhm):
    """輝線領域と輝線のない領域のマスクを返す

    輝線中心から ±(n_fwhm × FWHM) の内側を輝線領域、外側を輝線のない領域とする。
    """
    fwhm_ghz = velocity_width_to_freq_width(line_fwhm_kms, center_freq_ghz)
    in_line = np.abs(np.asarray(freq_ghz) - center_freq_ghz) <= n_fwhm * fwhm_ghz
    return in_line, ~in_line


def line_free_channels(line_free_wide, shifts_ch, start0, n_channels):
    """どの時刻でも輝線が入らない分光計チャンネルのマスク (n_channels,)"""
    shifts = np.unique(shifts_ch)
    ok = np.ones(n_channels, dtype=bool)
    for s in shifts:
        ok &= line_free_wide[start0 + s:start0 + s + n_channels]
    return ok


def integrated_intensity_kkms(spectrum_k, mask, channel_width_ghz, center_freq_ghz):
    """マスク内の積分強度 [K km/s]"""
    dv_kms = freq_width_to_velocity_width(channel_width_ghz, center_freq_ghz)
    return float(np.sum(spectrum_k[mask]) * dv_kms)


def time_power_spectrum(tod, dt_s, channel_mask=None):
    """時間方向のパワースペクトル密度(f = 0 とナイキスト周波数を除く)

    Returns
    -------
    freqs_hz : 周波数 [Hz]
    psd_per_channel : チャンネルごとのピリオドグラムの平均 [K^2/Hz]
    psd_common : チャンネル平均した時系列(共通成分)のピリオドグラム [K^2/Hz]
    """
    data = tod if channel_mask is None else tod[:, channel_mask]
    n_time = data.shape[0]
    freqs = np.fft.rfftfreq(n_time, dt_s)

    def periodogram(x):
        # 片側スペクトル。f = 0 とナイキスト周波数の成分は返さない
        return 2.0 * np.abs(np.fft.rfft(x, axis=0)) ** 2 * dt_s / n_time

    keep = slice(1, (n_time - 1) // 2 + 1)
    psd_per_channel = periodogram(data).mean(axis=1)
    psd_common = periodogram(data.mean(axis=1))
    return freqs[keep], psd_per_channel[keep], psd_common[keep]


def white_noise_psd_k2_per_hz(sigma_k, dt_s, n_average=1):
    """白色雑音(1 サンプルの標準偏差 sigma_k)の片側 PSD の期待値 [K^2/Hz]

    time_power_spectrum の規格化(2 |FFT|^2 Δt / N)では、分散 σ^2 の白色雑音は
    2 σ^2 Δt になる。n_average 本の独立なチャンネルを平均した時系列なら 1/n_average 倍。
    """
    return 2.0 * float(sigma_k) ** 2 * float(dt_s) / float(n_average)


def psd_comparison(tods: dict, dt_s, channel_mask, thermal_sigma_k, low_freq_hz=0.1):
    """除去前後の時系列の PSD を、理論的な熱雑音(白色)レベルと比べる

    tods : {'before': X, 'simple_pca': X - C, ...} の (n_time, n_channels) 配列 [K]
    channel_mask : PSD に使うチャンネル(輝線が一度も入らないチャンネル)
    low_freq_hz  : この周波数未満を「低周波域」として、白色レベルに対する超過を要約する

    Returns
    -------
    dict:
      'freqs_hz', 'white_per_channel', 'white_common'
      'psd'   : {label: (psd_per_channel, psd_common)}
      'excess': {label: {'low_freq_per_channel': 低周波域の平均 PSD / 白色レベル,
                         'low_freq_common': 同(チャンネル平均した時系列), ...}}
    比が 1 に近いほど、低周波の相関雑音が熱雑音のレベルまで落ちている。
    """
    n_used = int(np.count_nonzero(channel_mask)) if channel_mask is not None else None
    if n_used == 0:
        raise ValueError("PSD に使えるチャンネル(輝線が入らないチャンネル)がありません")
    out = {"psd": {}, "excess": {}}
    for label, tod in tods.items():
        freqs, per_ch, common = time_power_spectrum(tod, dt_s, channel_mask)
        out["psd"][label] = (per_ch, common)
        out["freqs_hz"] = freqs
    n_avg = n_used if n_used is not None else next(iter(tods.values())).shape[1]
    white_ch = white_noise_psd_k2_per_hz(thermal_sigma_k, dt_s)
    white_common = white_noise_psd_k2_per_hz(thermal_sigma_k, dt_s, n_avg)
    out["white_per_channel"], out["white_common"] = white_ch, white_common
    low = out["freqs_hz"] < low_freq_hz
    for label, (per_ch, common) in out["psd"].items():
        ex = {"low_freq_hz": float(low_freq_hz), "n_low_freq_bins": int(low.sum()),
              "all_freq_per_channel": float(per_ch.mean() / white_ch) if white_ch > 0 else None,
              "all_freq_common": float(common.mean() / white_common) if white_common > 0 else None}
        if low.any() and white_ch > 0:
            ex["low_freq_per_channel"] = float(per_ch[low].mean() / white_ch)
            ex["low_freq_common"] = float(common[low].mean() / white_common)
        else:
            ex["low_freq_per_channel"] = ex["low_freq_common"] = None
        out["excess"][label] = ex
    return out


def window_masks(freq_ghz, windows_ghz):
    """周波数窓 [(下限, 上限), ...] [GHz] の内側を輝線領域、外側を輝線のない領域とするマスク

    輝線の形が分からないキューブ(外部入力など)で、評価する領域を明示的に与えるためのもの。
    """
    freq = np.asarray(freq_ghz)
    in_line = np.zeros(freq.shape, dtype=bool)
    for lo, hi in windows_ghz:
        if not lo < hi:
            raise ValueError(f"周波数窓は 下限 < 上限 で指定してください: ({lo}, {hi})")
        in_line |= (freq >= lo) & (freq <= hi)
    return in_line, ~in_line


def evaluate(restored_k, truth_k, hits, n_time, freq_ghz, center_freq_ghz,
             line_fwhm_kms, n_fwhm, channel_width_ghz):
    """復元スペクトルと正解スペクトルを比べて評価値の辞書を返す

    評価は全時刻で観測された周波数範囲(hits == n_time)だけで行う。
    輝線領域は輝線中心から ±(n_fwhm × FWHM)(ガウシアン 1 本の内部生成キューブ用)。
    """
    in_line, line_free = line_masks(freq_ghz, center_freq_ghz, line_fwhm_kms, n_fwhm)
    return evaluate_masked(restored_k, truth_k, hits, n_time, in_line, line_free,
                           channel_width_ghz, center_freq_ghz)


def evaluate_masked(restored_k, truth_k, hits, n_time, in_line, line_free,
                    channel_width_ghz, ref_freq_ghz):
    """輝線領域・輝線のない領域のマスクを与えて評価する(evaluate の本体)

    ref_freq_ghz は積分強度 [K km/s] の速度換算に使う基準周波数。
    該当チャンネルが無い・正解が 0 などで定義できない評価値は None にする。
    """
    valid = hits == n_time
    in_line = in_line & valid
    line_free = line_free & valid
    residual = restored_k - truth_k
    has_free, has_valid = bool(line_free.any()), bool(valid.any())

    i_peak = int(np.argmax(np.where(valid, truth_k, -np.inf))) if has_valid else None
    peak_true = float(truth_k[i_peak]) if has_valid else None
    peak_rest = float(restored_k[i_peak]) if has_valid else None
    ii_true = integrated_intensity_kkms(truth_k, in_line, channel_width_ghz, ref_freq_ghz)
    ii_rest = integrated_intensity_kkms(restored_k, in_line, channel_width_ghz, ref_freq_ghz)
    return {
        "line_free_residual_rms_k": float(np.sqrt(np.mean(residual[line_free] ** 2))) if has_free else None,
        "line_free_residual_mean_k": float(np.mean(residual[line_free])) if has_free else None,
        "peak_true_k": peak_true,
        "peak_restored_k": peak_rest,
        "peak_ratio": float(restored_k[i_peak] / truth_k[i_peak]) if has_valid and peak_true != 0 else None,
        "integrated_true_kkms": ii_true,
        "integrated_restored_kkms": ii_rest,
        "integrated_ratio": ii_rest / ii_true if ii_true != 0 else None,
        "n_line_free_channels": int(line_free.sum()),
        "n_line_channels": int(in_line.sum()),
    }


def residual_summary(restored_k, truth_k, hits, n_time):
    """輝線領域の定義に依らない残差の要約(全時刻で観測された周波数だけで計算)"""
    valid = hits == n_time
    if not valid.any():
        return {"n_fully_observed_channels": 0, "residual_rms_k": None, "residual_max_abs_k": None}
    residual = (restored_k - truth_k)[valid]
    return {
        "n_fully_observed_channels": int(valid.sum()),
        "residual_rms_k": float(np.sqrt(np.mean(residual ** 2))),
        "residual_max_abs_k": float(np.max(np.abs(residual))),
    }


# ---------------------------------------------------------------------------
# スキャン(段階 2-B):キューブ・マップの評価
# ---------------------------------------------------------------------------

def moment0_map(cube, line_mask, channel_width_ghz, ref_freq_ghz):
    """輝線領域の積分強度マップ [K km/s]。NaN を含む画素は NaN"""
    dv_kms = freq_width_to_velocity_width(channel_width_ghz, ref_freq_ghz)
    return np.sum(np.asarray(cube)[:, :, line_mask], axis=2) * dv_kms


def peak_position(mom0, ra_axis, dec_axis, frac=0.5):
    """積分強度マップのピーク画素と、ピークの frac 倍以上の画素の強度重心 [arcsec]"""
    m = np.where(np.isfinite(mom0), mom0, -np.inf)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    peak = float(mom0[i, j])
    sel = np.isfinite(mom0) & (mom0 >= frac * peak)
    w = np.where(sel, mom0, 0.0)
    rr, dd = np.meshgrid(ra_axis, dec_axis, indexing="ij")
    total = w.sum()
    cen = (float((w * rr).sum() / total), float((w * dd).sum() / total)) if total > 0 else (None, None)
    return {"peak_kkms": peak, "peak_ra_arcsec": float(ra_axis[i]), "peak_dec_arcsec": float(dec_axis[j]),
            "centroid_ra_arcsec": cen[0], "centroid_dec_arcsec": cen[1]}


def evaluate_scan_cube(restored, reference, truth, spatial_mask, freq_valid, in_line, line_free,
                       ra_axis, dec_axis, channel_width_ghz, ref_freq_ghz, pixel_arcsec,
                       source_ra_arcsec, source_dec_arcsec):
    """スキャンで再構成したキューブを評価する

    restored  : 再構成キューブ (n_ra, n_dec, n_wide) [K](NaN = データなし)
    reference : 天体信号だけ(雑音なし・除去なし)の時系列を同じ手順でグリッディングしたキューブ。
                グリッディングによる平滑化を含むので、雑音・除去の誤差だけを測る基準になる
    truth     : 正解キューブ(ビーム畳み込み済み)を出力グリッドの画素中心でバイリニア補間したもの
    spatial_mask : 評価する画素 (n_ra, n_dec)(スキャン領域の内側)
    freq_valid   : 全サンプルで観測された周波数 (n_wide,)
    in_line / line_free : 輝線領域・輝線のない領域 (n_wide,)
    """
    restored, reference, truth = (np.asarray(a, dtype=float) for a in (restored, reference, truth))
    vox = spatial_mask[:, :, None] & freq_valid[None, None, :] & np.isfinite(restored)
    vox_free = vox & line_free[None, None, :]
    line = in_line & freq_valid

    def rms(a, m):
        return float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else None

    mom = {name: moment0_map(c, line, channel_width_ghz, ref_freq_ghz)
           for name, c in (("restored", restored), ("reference", reference), ("truth", truth))}
    sm = spatial_mask & np.isfinite(mom["restored"])
    area = pixel_arcsec ** 2

    def total_flux(m):
        return float(np.sum(m[sm]) * area)

    flux = {k: total_flux(v) for k, v in mom.items()}
    pos = peak_position(np.where(sm, mom["restored"], np.nan), ra_axis, dec_axis)
    pos_true = peak_position(np.where(sm, mom["reference"], np.nan), ra_axis, dec_axis)
    i_pk = np.unravel_index(int(np.nanargmax(np.where(vox, reference, np.nan))), reference.shape)
    return {
        "cube_rmse_vs_reference_k": rms(restored - reference, vox),
        "cube_rmse_vs_truth_k": rms(restored - truth, vox),
        "cube_line_free_rms_k": rms(restored - reference, vox_free),
        "voxel_peak_reference_k": float(reference[i_pk]),
        "voxel_peak_restored_k": float(restored[i_pk]),
        "voxel_peak_ratio": float(restored[i_pk] / reference[i_pk]) if reference[i_pk] != 0 else None,
        "mom0_residual_rms_kkms": rms(mom["restored"] - mom["reference"], sm),
        "mom0_peak_restored_kkms": pos["peak_kkms"],
        "mom0_peak_reference_kkms": pos_true["peak_kkms"],
        "mom0_peak_ratio": pos["peak_kkms"] / pos_true["peak_kkms"] if pos_true["peak_kkms"] else None,
        "peak_position_error_arcsec": float(np.hypot(pos["peak_ra_arcsec"] - source_ra_arcsec,
                                                     pos["peak_dec_arcsec"] - source_dec_arcsec)),
        "centroid_error_arcsec": (float(np.hypot(pos["centroid_ra_arcsec"] - source_ra_arcsec,
                                                 pos["centroid_dec_arcsec"] - source_dec_arcsec))
                                  if pos["centroid_ra_arcsec"] is not None else None),
        "centroid_ra_arcsec": pos["centroid_ra_arcsec"],
        "centroid_dec_arcsec": pos["centroid_dec_arcsec"],
        "total_flux_restored_kkms_arcsec2": flux["restored"],
        "total_flux_reference_kkms_arcsec2": flux["reference"],
        "total_flux_truth_kkms_arcsec2": flux["truth"],
        "flux_recovery_vs_reference": flux["restored"] / flux["reference"] if flux["reference"] else None,
        "flux_recovery_vs_truth": flux["restored"] / flux["truth"] if flux["truth"] else None,
        "n_eval_pixels": int(sm.sum()),
        "n_eval_voxels": int(vox.sum()),
    }
