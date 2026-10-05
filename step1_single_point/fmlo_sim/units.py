"""単位変換(FWHM↔σ、速度↔周波数)"""
import numpy as np

C_KMS = 299792.458  # 光速 [km/s]

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


def fwhm_to_sigma(fwhm):
    """ガウシアンの FWHM を σ に変換する"""
    return fwhm / FWHM_PER_SIGMA


def sigma_to_fwhm(sigma):
    """ガウシアンの σ を FWHM に変換する"""
    return sigma * FWHM_PER_SIGMA


def velocity_width_to_freq_width(dv_kms, nu):
    """速度幅 [km/s] を周波数幅に変換する(Δν = ν Δv / c)

    戻り値の単位は nu と同じ(nu が GHz なら GHz)。
    """
    return nu * dv_kms / C_KMS


def freq_width_to_velocity_width(dnu, nu):
    """周波数幅を速度幅 [km/s] に変換する(Δv = c Δν / ν)。dnu と nu は同じ単位"""
    return C_KMS * dnu / nu


def freq_to_velocity(nu, nu0):
    """周波数を、nu0 を基準とした視線速度 [km/s] に変換する(radio 定義)"""
    return C_KMS * (nu0 - nu) / nu0
