"""論文 Fig. 6 のパネル画像を数値化し、シミュレーション結果と比較する

入力は arXiv:1911.02574 のソース (https://arxiv.org/e-print/1911.02574) に含まれる
flowchart.pdf。リポジトリには同梱していないので、各自で取得してパスを渡す。
PyMuPDF が必要 (pip install pymupdf)。

    python tools/compare_taniguchi2020_fig6.py path/to/flowchart.pdf
    python tools/compare_taniguchi2020_fig6.py path/to/flowchart.pdf --write-profile   # CSV を作り直す

数値化の方法
- PDF に埋め込まれた 8 枚のパネル (720x720, JPEG, viridis) を取り出し、各画素を
  viridis の最も近い色に対応させて 0–1 の値に戻す。
- T^ast と E^nc のパネルの背景 (値 0) が 0.2275、2 回目の E^nc の輝線なし領域の
  標準偏差が 0.1361 なので、(v − 0.2275) / 0.1361 で雑音 σ 単位に直す。
- 縦 7.2 画素が 1 時刻。境界を避けて行平均し (100, 720) にする。下端が n=0。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fmlo_sim.taniguchi2020 import load_fig6_config, run_fig6, to_panel_pixels  # noqa: E402

ZERO, SIGMA = 0.2275, 0.1361
# パネル名 → 図上の位置 (列, 行)。PDF 内の画像は bbox の位置で識別する
PANELS = {(0, 0): "x_in1", (1, 0): "t_cor1", (2, 0): "t_ast1", (3, 0): "e_nc1",
          (0, 1): "x_in2", (1, 1): "t_cor2", (2, 1): "t_ast2", (3, 1): "e_nc2"}
LINE_FREE = np.r_[0:200, 560:720]


def digitize(pdf_path):
    import pymupdf
    from matplotlib import colormaps
    lut = colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    out = {}
    for info in page.get_image_info(xrefs=True):
        if info["width"] != 720 or info["height"] != 720:
            continue
        x0, y0 = info["bbox"][:2]
        key = (int(round((x0 - 95) / 240)), int(round((y0 - 13) / 225)))
        pix = pymupdf.Pixmap(doc, info["xref"])
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(720, 720, pix.n)[:, :, :3].astype(float)
        idx = ((rgb[:, :, None, :] - lut[None, None]) ** 2).sum(-1).argmin(-1)
        val = (idx / 255.0 - ZERO) / SIGMA
        rows = []
        for i in range(100):
            lo, hi = int(np.ceil(i * 7.2)) + 1, int(np.floor((i + 1) * 7.2)) - 1
            rows.append(val[lo:hi + 1].mean(0))
        out[PANELS[key]] = np.array(rows)[::-1]
    if len(out) != 8:
        raise RuntimeError(f"720x720 のパネルが 8 枚見つかりません ({len(out)} 枚)")
    return out


def line_track(t_ast):
    """各時刻の輝線中心 [px](強度重心)"""
    x = np.arange(t_ast.shape[1])
    w = np.clip(t_ast, 0, None)
    return (w * x).sum(1) / w.sum(1)


def aligned_profile(img, pos, half=40):
    x = np.arange(img.shape[1])
    off = np.arange(-half, half + 1)
    return np.mean([np.interp(off + p, x, r) for r, p in zip(img, pos)], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--config", default=str(ROOT / "config" / "taniguchi2020_fig6.yaml"))
    ap.add_argument("--write-profile", action="store_true",
                    help="相関成分の時間プロファイル CSV を数値化結果で上書きする")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n-seeds", type=int, default=8, help="雑音の実現を変えて平均する数")
    a = ap.parse_args()

    paper = digitize(a.pdf)
    cfg = load_fig6_config(a.config, dict(o.split('=', 1) for o in a.overrides))

    prof = paper["t_cor2"][:, LINE_FREE].mean(1)
    if a.write_profile:
        path = ROOT / cfg.correlated.profile_file
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Taniguchi et al. (2020) Fig.6 の相関成分 T^cor の時間プロファイル"
                    "(tools/compare_taniguchi2020_fig6.py --write-profile で生成)\n")
            f.write("# 単位: 非相関雑音の標準偏差 (E^nc ~ N(0,1))。n=0 が最初の時刻(パネル下端)\n# n,value\n")
            for i, v in enumerate(prof):
                f.write(f"{i},{v:.3f}\n")
        print("wrote", path)

    pos = line_track(paper["t_ast2"])
    sims = {k: [] for k in paper}
    for seed in range(a.n_seeds):
        cfg.seed = seed
        _, res = run_fig6(cfg)
        for i, st in enumerate(res.steps[:2], 1):
            for name in ("x_in", "t_cor", "t_ast", "e_nc"):
                # 論文画像はカラーレンジで頭打ちになっているので同じ範囲に切る
                sims[f"{name}{i}"].append(np.clip(to_panel_pixels(getattr(st, name), 720),
                                                  cfg.plot.vmin, cfg.plot.vmax))

    print(f"\n輝線に沿って平均した断面 (±40 px) の比較。単位は雑音 σ。シミュレーションは {a.n_seeds} 実現の平均")
    print(f"{'panel':8s}{'RMS差':>8s}{'論文 max':>10s}{'sim max':>9s}{'論文 min':>10s}{'sim min':>9s}")
    for name in ("x_in1", "t_cor1", "t_ast1", "e_nc1", "x_in2", "t_cor2", "t_ast2", "e_nc2"):
        p = aligned_profile(paper[name], pos)
        s = np.mean([aligned_profile(m, pos) for m in sims[name]], axis=0)
        print(f"{name:8s}{np.sqrt(np.mean((p - s) ** 2)):8.2f}{p.max():10.2f}{s.max():9.2f}{p.min():10.2f}{s.min():9.2f}")

    print("\n輝線なし領域 (px 0–199, 560–719) の統計")
    print(f"{'panel':8s}{'論文 mean':>10s}{'sim mean':>10s}{'論文 std':>10s}{'sim std':>9s}")
    for name in ("x_in1", "e_nc1", "e_nc2"):
        p = paper[name][:, LINE_FREE]
        s = np.concatenate([m[:, LINE_FREE] for m in sims[name]])
        print(f"{name:8s}{p.mean():10.2f}{s.mean():10.2f}{p.std():10.2f}{s.std():9.2f}")
    sim_track = line_track(np.mean(sims["t_ast2"], axis=0))
    print(f"\n輝線位置のずれ (sim − 論文): 平均 {np.mean(sim_track - pos):+.2f} px, "
          f"最大 |差| {np.max(np.abs(sim_track - pos)):.2f} px (パネル幅 720 px)")


if __name__ == "__main__":
    main()
