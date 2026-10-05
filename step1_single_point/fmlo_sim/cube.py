"""正解キューブ(RA × Dec × 周波数)の生成・保存・スペクトル取り出し"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .config import Config
from .units import fwhm_to_sigma, velocity_width_to_freq_width


@dataclass
class FrequencyGrid:
    """分光計の帯域より広い周波数グリッド

    FMLO で観測窓をずらせるよう、帯域の両側に (変調の半幅 + 余白) だけ広く取る。
    チャンネル幅は分光計と同じなので、窓の切り出しは整数チャンネルのスライスになる。
    """

    freq_ghz: np.ndarray       # 広いグリッドの周波数 [GHz]、長さ n_wide
    n_channels: int            # 分光計のチャンネル数
    start0: int                # シフト量 0 のときの窓の先頭インデックス
    channel_width_ghz: float   # チャンネル幅 [GHz]
    center_freq_ghz: float     # シフト量 0 のときの帯域中心 [GHz]

    @property
    def n_wide(self) -> int:
        return self.freq_ghz.size

    @property
    def bandwidth_ghz(self) -> float:
        return self.n_channels * self.channel_width_ghz


def make_frequency_grid(cfg: Config) -> FrequencyGrid:
    n_ch = cfg.spectrometer.n_channels
    dnu = cfg.spectrometer.channel_width_mhz * 1e-3
    start0 = cfg.modulation.half_width_ch + cfg.cube.margin_ch
    n_wide = n_ch + 2 * start0
    # シフト量 0 の窓の中心が spectrometer.center_freq_ghz になるように置く
    offset_ch = np.arange(n_wide) - start0 - (n_ch - 1) / 2.0
    freq = cfg.spectrometer.center_freq_ghz + offset_ch * dnu
    return FrequencyGrid(freq, n_ch, start0, dnu, cfg.spectrometer.center_freq_ghz)


def make_spatial_axes(cfg: Config):
    """RA, Dec 軸 [arcsec](マップ中心からのオフセット)"""
    half = cfg.cube.half_extent_arcsec
    n_half = int(round(half / cfg.cube.pixel_arcsec))
    axis = np.arange(-n_half, n_half + 1) * cfg.cube.pixel_arcsec
    return axis.copy(), axis.copy()


def make_gaussian_cube(cfg: Config, grid: FrequencyGrid):
    """ガウシアンのボール1つの正解キューブを作る

    Returns
    -------
    cube : (n_ra, n_dec, n_freq) の配列 [K]。cube[i_ra, i_dec, i_freq]
    ra_arcsec, dec_arcsec : 空間軸 [arcsec]
    """
    src = cfg.source
    ra, dec = make_spatial_axes(cfg)
    sigma_freq_ghz = fwhm_to_sigma(
        velocity_width_to_freq_width(src.line_fwhm_kms, src.center_freq_ghz)
    )
    prof_ra = np.exp(-0.5 * ((ra - src.center_ra_arcsec) / src.sigma_arcsec) ** 2)
    prof_dec = np.exp(-0.5 * ((dec - src.center_dec_arcsec) / src.sigma_arcsec) ** 2)
    prof_freq = np.exp(-0.5 * ((grid.freq_ghz - src.center_freq_ghz) / sigma_freq_ghz) ** 2)
    cube = src.peak_k * np.einsum("i,j,k->ijk", prof_ra, prof_dec, prof_freq)
    return cube.astype(np.float32), ra, dec


def save_cube(path, cube, ra_arcsec, dec_arcsec, freq_ghz, cfg: Config) -> None:
    """キューブを .npz に保存する(段階2以降で再利用する)"""
    meta = {
        "axes": "cube[i_ra, i_dec, i_freq]",
        "units": {"cube": "K", "ra": "arcsec", "dec": "arcsec", "freq": "GHz"},
        "config": cfg.to_dict(),
    }
    np.savez(
        path,
        cube=cube,
        ra_arcsec=ra_arcsec,
        dec_arcsec=dec_arcsec,
        freq_ghz=freq_ghz,
        meta=json.dumps(meta, ensure_ascii=False),
    )


def load_cube(path):
    """save_cube で保存したキューブを読む。(cube, ra, dec, freq, meta) を返す"""
    with np.load(path) as f:
        return (
            f["cube"],
            f["ra_arcsec"],
            f["dec_arcsec"],
            f["freq_ghz"],
            json.loads(str(f["meta"])),
        )


def extract_spectrum(cube, ra_axis, dec_axis, ra_arcsec, dec_arcsec):
    """指定位置に最も近いピクセルのスペクトル [K] を取り出す

    Returns
    -------
    spectrum : 長さ n_freq の配列 [K]
    (i_ra, i_dec) : 取り出したピクセルのインデックス
    """
    i_ra = int(np.argmin(np.abs(ra_axis - ra_arcsec)))
    i_dec = int(np.argmin(np.abs(dec_axis - dec_arcsec)))
    return cube[i_ra, i_dec, :].astype(np.float64), (i_ra, i_dec)
