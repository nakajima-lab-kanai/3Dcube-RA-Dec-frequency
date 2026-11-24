"""正解キューブ(RA × Dec × 周波数)の生成・保存・スペクトル取り出し"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

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
    # cycle_jitter が無効なら max_abs_shift_ch = half_width_ch(従来と同じグリッド)
    start0 = cfg.modulation.max_abs_shift_ch + cfg.cube.margin_ch
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


def convolve_beam(cube, pixel_arcsec, beam_fwhm_arcsec, truncate=4.0):
    """空間 2 軸をガウシアンビーム(FWHM 指定)で畳み込む

    ビームは面積 1 に規格化する(輝度温度 [K] のまま、点源の総フラックスを保つ)。
    マップの外側の空は 0 K とみなす(mode='constant')。周波数軸は畳み込まない。
    ピクセルがビームに比べて粗いと離散化の誤差が大きくなるので、
    beam_fwhm / pixel < 2 ならエラーにする。
    """
    from scipy.ndimage import gaussian_filter

    sigma_pix = fwhm_to_sigma(beam_fwhm_arcsec) / pixel_arcsec
    if beam_fwhm_arcsec / pixel_arcsec < 2.0:
        raise ValueError(
            f"ビーム FWHM {beam_fwhm_arcsec} arcsec に対してピクセル {pixel_arcsec} arcsec が粗すぎます"
            "(FWHM / pixel >= 2 にしてください)"
        )
    out = gaussian_filter(np.asarray(cube, dtype=np.float64), sigma=(sigma_pix, sigma_pix, 0.0),
                          mode="constant", cval=0.0, truncate=truncate)
    return out.astype(np.asarray(cube).dtype)


def bilinear_matrix(ra_axis, dec_axis, ra_arcsec, dec_arcsec):
    """各サンプル位置で (RA, Dec) 画素をバイリニア補間する疎行列 (n_sample × n_ra*n_dec)

    行列を cube.reshape(n_ra*n_dec, n_freq) に掛けると、各位置のスペクトルになる。
    軸は昇順・等間隔。位置が軸の範囲 [最初の画素中心, 最後の画素中心] の外ならエラー
    (外挿・端の画素への丸めはしない)。
    """
    from scipy import sparse

    ra_axis = np.asarray(ra_axis, dtype=float)
    dec_axis = np.asarray(dec_axis, dtype=float)
    ra = np.atleast_1d(np.asarray(ra_arcsec, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_arcsec, dtype=float))
    n_ra, n_dec = ra_axis.size, dec_axis.size

    def frac_index(axis, x, name):
        if axis.size == 1:
            if np.any(np.abs(x - axis[0]) > 1e-9):
                raise ValueError(f"{name} 軸が 1 画素だけなので、位置は {axis[0]} arcsec に限ります")
            return np.zeros(x.size, dtype=int), np.zeros(x.size)
        pix = (axis[-1] - axis[0]) / (axis.size - 1)
        f = (x - axis[0]) / pix
        tol = 1e-9
        if np.any(f < -tol) or np.any(f > axis.size - 1 + tol):
            raise ValueError(
                f"{name} 位置 [{x.min()}, {x.max()}] arcsec がキューブの範囲 "
                f"[{axis[0]}, {axis[-1]}] arcsec の外です(外挿は行いません)"
            )
        f = np.clip(f, 0.0, axis.size - 1)
        i0 = np.minimum(np.floor(f).astype(int), axis.size - 2)
        return i0, f - i0

    i0, fr = frac_index(ra_axis, ra, "RA")
    j0, fd = frac_index(dec_axis, dec, "Dec")
    rows, cols, vals = [], [], []
    n = ra.size
    for di, wi in ((0, 1.0 - fr), (1, fr)):
        for dj, wj in ((0, 1.0 - fd), (1, fd)):
            ii = np.minimum(i0 + di, n_ra - 1)
            jj = np.minimum(j0 + dj, n_dec - 1)
            rows.append(np.arange(n))
            cols.append(ii * n_dec + jj)
            vals.append(wi * wj)
    mat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n_ra * n_dec)
    )
    mat.sum_duplicates()
    return mat


def sample_spectra(cube, ra_axis, dec_axis, ra_arcsec, dec_arcsec):
    """各位置のスペクトルをバイリニア補間で取り出す。(n_sample, n_freq) [K]"""
    cube = np.asarray(cube)
    mat = bilinear_matrix(ra_axis, dec_axis, ra_arcsec, dec_arcsec)
    flat = cube.reshape(-1, cube.shape[2]).astype(np.float64)
    return np.asarray(mat @ flat)


CUBE_AXES = "cube[i_ra, i_dec, i_freq]"
CUBE_UNITS = {"cube": "K", "ra": "arcsec", "dec": "arcsec", "freq": "GHz"}


def save_cube(path, cube, ra_arcsec, dec_arcsec, freq_ghz, cfg: Optional[Config] = None) -> None:
    """キューブを .npz に保存する(段階2以降で再利用する)

    cfg は内部生成キューブの生成条件の記録用。外部で作ったキューブなら None でよい
    (meta の config は null になる)。
    """
    meta = {
        "axes": CUBE_AXES,
        "units": dict(CUBE_UNITS),
        "config": None if cfg is None else cfg.to_dict(),
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
