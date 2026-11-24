"""タイムストリーム ⇄ 3 次元キューブ(段階 2-B)

- グリッディング(cube making):復調した時系列(観測していない周波数は NaN、重み W^NaN)と
  指向位置から、ガウシアンカーネルの重み付き平均で出力キューブ (RA × Dec × 周波数) を作る
- 逆投影(projection):キューブからスキャン軌跡・変調パターンに従ってモデル時系列を作る
  (反復 PCA の輝線モデル A^(i) に使う)
- basket weave:x_scan と y_scan のキューブを統合する(重み付き平均 / フーリエ空間の重み付け)

出力キューブの値は、画素ごと・周波数ごとに
    T(p, ν) = Σ_t w(p, t) W(t, ν) d(t, ν) / Σ_t w(p, t) W(t, ν)
ここで w はカーネルの重み、W は復調の有効サンプルマスク。分母が 0 なら NaN。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import cube as cube_mod
from . import modulation, removal
from .units import fwhm_to_sigma


@dataclass
class MapGrid:
    """出力キューブの空間グリッド(画素中心の座標 [arcsec]、昇順・等間隔)"""

    ra_arcsec: np.ndarray
    dec_arcsec: np.ndarray
    pixel_arcsec: float

    @property
    def shape(self):
        return (self.ra_arcsec.size, self.dec_arcsec.size)

    def pixel_centers(self):
        rr, dd = np.meshgrid(self.ra_arcsec, self.dec_arcsec, indexing="ij")
        return rr.ravel(), dd.ravel()


def make_map_grid(cfg) -> MapGrid:
    """スキャン領域を覆う出力グリッド。半幅は map_size/2 を画素の整数倍に切り上げる"""
    pix = cfg.mapping.grid_pixel_arcsec
    half = cfg.scan.map_size_arcsec / 2.0 if cfg.scan.mode != "single_point" else 0.0
    n_half = int(np.ceil(half / pix - 1e-9))
    axis = np.arange(-n_half, n_half + 1) * pix
    return MapGrid(axis + cfg.scan.center_ra_arcsec, axis + cfg.scan.center_dec_arcsec, pix)


def gridding_matrix(map_grid: MapGrid, ra_arcsec, dec_arcsec, kernel_fwhm_arcsec, support_fwhm=1.5):
    """カーネルの重み w(p, t) の疎行列 (n_pix × n_time)

    ガウシアンカーネル exp(-r^2 / 2σ^2)(σ は kernel_fwhm から)を、
    r <= support_fwhm × kernel_fwhm で打ち切る。
    """
    ra, dec = np.asarray(ra_arcsec, float), np.asarray(dec_arcsec, float)
    pix = map_grid.pixel_arcsec
    sigma = fwhm_to_sigma(kernel_fwhm_arcsec)
    r_max = support_fwhm * kernel_fwhm_arcsec
    n_ra, n_dec = map_grid.shape
    ci = np.rint((ra - map_grid.ra_arcsec[0]) / pix).astype(int)
    cj = np.rint((dec - map_grid.dec_arcsec[0]) / pix).astype(int)
    reach = int(np.ceil(r_max / pix)) + 1
    rows, cols, vals = [], [], []
    t_idx = np.arange(ra.size)
    for di in range(-reach, reach + 1):
        ii = ci + di
        ok_i = (ii >= 0) & (ii < n_ra)
        for dj in range(-reach, reach + 1):
            jj = cj + dj
            ok = ok_i & (jj >= 0) & (jj < n_dec)
            if not ok.any():
                continue
            r2 = ((map_grid.ra_arcsec[np.clip(ii, 0, n_ra - 1)] - ra) ** 2
                  + (map_grid.dec_arcsec[np.clip(jj, 0, n_dec - 1)] - dec) ** 2)
            ok &= r2 <= r_max ** 2
            rows.append(ii[ok] * n_dec + jj[ok])
            cols.append(t_idx[ok])
            vals.append(np.exp(-0.5 * r2[ok] / sigma ** 2))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n_ra * n_dec, ra.size))


@dataclass
class GriddedCube:
    """グリッディングの結果。分子・分母を持っておくと、パスの統合が単純な足し算になる"""

    numerator: np.ndarray     # Σ w W d  (n_ra, n_dec, n_wide) [K]
    weight: np.ndarray        # Σ w W    (n_ra, n_dec, n_wide)
    weight_sq: np.ndarray     # Σ w^2 W  (n_ra, n_dec, n_wide)(雑音の見積もり用)

    @property
    def data(self) -> np.ndarray:
        """重み付き平均のキューブ [K]。データのない画素・周波数は NaN"""
        out = np.full(self.numerator.shape, np.nan)
        np.divide(self.numerator, self.weight, out=out, where=self.weight > 0)
        return out

    def noise_scale(self) -> np.ndarray:
        """1 サンプルの雑音が σ のときの各ボクセルの雑音 / σ = sqrt(Σw^2) / Σw"""
        out = np.full(self.weight.shape, np.nan)
        np.divide(np.sqrt(self.weight_sq), self.weight, out=out, where=self.weight > 0)
        return out

    def __add__(self, other: "GriddedCube") -> "GriddedCube":
        return GriddedCube(self.numerator + other.numerator, self.weight + other.weight,
                           self.weight_sq + other.weight_sq)


def grid_tod(tod, shifts_ch, start0, n_wide, wmat, map_shape) -> GriddedCube:
    """1 パスの時系列(分光器チャンネル)を復調し、出力キューブにグリッディングする"""
    demod, w_nan = modulation.demodulate_tod(tod, shifts_ch, start0, n_wide)
    d0 = np.where(w_nan > 0, demod, 0.0)
    m = w_nan.astype(np.float64)
    num = np.asarray(wmat @ d0)
    den = np.asarray(wmat @ m)
    den2 = np.asarray(wmat.multiply(wmat) @ m)
    shape = (*map_shape, n_wide)
    return GriddedCube(num.reshape(shape), den.reshape(shape), den2.reshape(shape))


def project_cube(cube, map_grid: MapGrid, ra_arcsec, dec_arcsec, shifts_ch, start0, n_channels):
    """キューブを軌跡に沿ってバイリニア補間し、各時刻の観測窓を切り出したモデル時系列

    NaN(データのないボクセル)は 0 として扱う。Returns (n_time, n_channels) [K]
    """
    cube = np.nan_to_num(np.asarray(cube, dtype=float), nan=0.0)
    wide = cube_mod.sample_spectra(cube, map_grid.ra_arcsec, map_grid.dec_arcsec, ra_arcsec, dec_arcsec)
    return modulation.modulate(wide, shifts_ch, start0, n_channels)


def robust_sigma(x):
    """1.4826 × MAD(中央値絶対偏差)による雑音の標準偏差"""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def cutoff_cube(gridded: GriddedCube, sample_sigma_k, cutoff_sigma, smooth_ch=1):
    """σ-cutoff:|T| > cutoff_sigma × σ_voxel のボクセルだけ残したキューブ(それ以外は 0)

    σ_voxel = sample_sigma × sqrt(Σw^2)/Σw(独立なサンプルの重み付き平均の雑音)。
    smooth_ch > 1 なら、先にキューブを周波数方向に smooth_ch チャンネルの箱型で平滑化し、
    雑音も 1/sqrt(smooth_ch) 倍として判定する(残すのは平滑化した値)。スキャンでは
    1 ボクセルの S/N が低く、広がった天体の裾が判定で落ちやすいので、それを補うためのもの。
    """
    data = gridded.data
    sig = sample_sigma_k * gridded.noise_scale()
    if smooth_ch > 1:
        from scipy.ndimage import uniform_filter1d

        finite = np.isfinite(data)
        data = np.where(finite, uniform_filter1d(np.where(finite, data, 0.0), int(smooth_ch), axis=2,
                                                 mode="nearest"), np.nan)
        sig = sig / np.sqrt(smooth_ch)
    keep = np.isfinite(data) & (np.abs(data) > cutoff_sigma * sig)
    return np.where(keep, data, 0.0)


# ---------------------------------------------------------------------------
# basket weave
# ---------------------------------------------------------------------------

def combine_weighted_mean(cubes) -> GriddedCube:
    """パスごとのグリッディング結果を重み(Σ w W)付きで平均する(分子・分母の和)"""
    total = cubes[0]
    for c in cubes[1:]:
        total = total + c
    return total


def combine_fourier(cube_x, cube_y):
    """フーリエ空間の重み付けによる basket weave(Emerson & Gräve 1988 の考え方)

    x_scan(RA 方向に走査)の走査方向の雑音(行ごとのオフセットなど)は Dec だけの関数になり、
    フーリエ空間で k_RA ≈ 0 の軸に集まる。そこで
        w_x(k) = k_RA^2 / |k|^2、 w_y(k) = k_Dec^2 / |k|^2(k = 0 は 1/2 ずつ)
    で重み付けして足す。w_x + w_y = 1 なので、両方のマップに共通な天体の構造は保たれる。
    NaN は 0 で埋めて変換し、両方のマップで NaN のボクセルは NaN に戻す。
    """
    x = np.asarray(cube_x, dtype=float)
    y = np.asarray(cube_y, dtype=float)
    both_nan = ~np.isfinite(x) & ~np.isfinite(y)
    x = np.where(np.isfinite(x), x, np.where(np.isfinite(y), y, 0.0))
    y = np.where(np.isfinite(y), y, x)
    k_ra = np.fft.fftfreq(x.shape[0])[:, None]
    k_dec = np.fft.fftfreq(x.shape[1])[None, :]
    k2 = k_ra ** 2 + k_dec ** 2
    wx = np.divide(k_ra ** 2, k2, out=np.full(k2.shape, 0.5), where=k2 > 0)
    wy = 1.0 - wx
    fx = np.fft.fft2(x, axes=(0, 1))
    fy = np.fft.fft2(y, axes=(0, 1))
    out = np.real(np.fft.ifft2(wx[:, :, None] * fx + wy[:, :, None] * fy, axes=(0, 1)))
    out[both_nan] = np.nan
    return out


# ---------------------------------------------------------------------------
# スキャンの雑音除去(simple_pca / iterative_pca)→ グリッディング
# ---------------------------------------------------------------------------

@dataclass
class PassGeometry:
    """パスごとに使い回す量(軌跡・シフト量・カーネルの重み行列)"""

    name: str
    ra_arcsec: np.ndarray
    dec_arcsec: np.ndarray
    shifts_ch: np.ndarray
    wmat: sparse.csr_matrix


@dataclass
class ScanReduction:
    """1 組のパス(x だけ・y だけ・x と y)の除去 → グリッディングの結果"""

    label: str
    method: str
    gridded: GriddedCube              # 全パスの重み付き平均(分子・分母)
    gridded_per_pass: list            # パスごとの GriddedCube
    cleaned: list                     # パスごとの X - C [K]
    removed: list                     # パスごとの C [K]
    model_tod: list                   # パスごとの A(iterative_pca のみ。simple_pca は 0)
    pca: list                         # パスごとの PcaResult(最終回)
    iterative: removal.IterativePcaResult = None


def scan_model_fn(geoms, map_grid, start0, n_wide, n_channels, cutoff_sigma, smooth_ch=1):
    """スキャンの輝線モデル:全パスをまとめてグリッディング → σ-cutoff → 各パスへ逆投影

    σ-cutoff の 1 サンプルの雑音は、X - C の時系列全体の MAD から見積もる
    (輝線が入るサンプルは一部なので、MAD はほぼ雑音で決まる)。
    """

    def model_fn(cleaned_list):
        cubes = [grid_tod(c, g.shifts_ch, start0, n_wide, g.wmat, map_grid.shape)
                 for c, g in zip(cleaned_list, geoms)]
        total = combine_weighted_mean(cubes)
        sigma = robust_sigma(np.concatenate([c.ravel() for c in cleaned_list]))
        model_cube = cutoff_cube(total, sigma, cutoff_sigma, smooth_ch)
        return [project_cube(model_cube, map_grid, g.ra_arcsec, g.dec_arcsec, g.shifts_ch, start0, n_channels)
                for g in geoms]

    return model_fn


def reduce_passes(label, observed_list, geoms, map_grid, start0, n_wide, removal_cfg) -> ScanReduction:
    """観測時系列(パスごと)から雑音を除去し、出力キューブにグリッディングする(正解は使わない)

    removal_cfg.method:
      simple_pca    : パスごとに PCA を 1 回(X - C をグリッディング)
      iterative_pca : 輝線モデル(全パスのキューブから逆投影)を引きながら PCA をくり返す
    """
    r = removal_cfg
    n_ch = observed_list[0].shape[1]
    if r.method == "simple_pca":
        pcas = [removal.pca_remove(x, r.n_components, r.mean_mode) for x in observed_list]
        cleaned = [p.cleaned for p in pcas]
        removed = [p.removed for p in pcas]
        model = [np.zeros_like(x) for x in observed_list]
        it = None
    elif r.method == "iterative_pca":
        it = removal.iterative_pca(observed_list, r.n_components, r.mean_mode,
                                   scan_model_fn(geoms, map_grid, start0, n_wide, n_ch, r.cutoff_sigma,
                                                 r.model_smooth_ch),
                                   r.max_iter, r.tol)
        pcas, cleaned, removed, model = it.pca, it.cleaned, it.removed, it.model_tod
    else:
        raise ValueError(f"未知の removal.method です: {r.method}")
    per_pass = [grid_tod(c, g.shifts_ch, start0, n_wide, g.wmat, map_grid.shape)
                for c, g in zip(cleaned, geoms)]
    return ScanReduction(label, r.method, combine_weighted_mean(per_pass), per_pass,
                         cleaned, removed, model, pcas, it)
