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


def evaluate(restored_k, truth_k, hits, n_time, freq_ghz, center_freq_ghz,
             line_fwhm_kms, n_fwhm, channel_width_ghz):
    """復元スペクトルと正解スペクトルを比べて評価値の辞書を返す

    評価は全時刻で観測された周波数範囲(hits == n_time)だけで行う。
    """
    valid = hits == n_time
    in_line, line_free = line_masks(freq_ghz, center_freq_ghz, line_fwhm_kms, n_fwhm)
    in_line &= valid
    line_free &= valid
    residual = restored_k - truth_k

    i_peak = int(np.argmax(np.where(valid, truth_k, -np.inf)))
    ii_true = integrated_intensity_kkms(truth_k, in_line, channel_width_ghz, center_freq_ghz)
    ii_rest = integrated_intensity_kkms(restored_k, in_line, channel_width_ghz, center_freq_ghz)
    return {
        "line_free_residual_rms_k": float(np.sqrt(np.mean(residual[line_free] ** 2))),
        "line_free_residual_mean_k": float(np.mean(residual[line_free])),
        "peak_true_k": float(truth_k[i_peak]),
        "peak_restored_k": float(restored_k[i_peak]),
        "peak_ratio": float(restored_k[i_peak] / truth_k[i_peak]),
        "integrated_true_kkms": ii_true,
        "integrated_restored_kkms": ii_rest,
        "integrated_ratio": ii_rest / ii_true,
        "n_line_free_channels": int(line_free.sum()),
        "n_line_channels": int(in_line.sum()),
    }
