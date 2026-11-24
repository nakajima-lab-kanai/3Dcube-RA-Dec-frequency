"""
A. キューブ(RA × Dec × 周波数)と観測設定から固定点の模擬観測を生成・保存する

入力キューブは cube.save_cube の形式(.npz。run_step1.py が出力する cube.npz と同じ)。
解析は行わない。保存した観測データは run_analyze_observation.py で解析・評価する。

使い方:
    # 既存の出力キューブを外部入力として観測するpython run_observe_cube.py --cube outputs/step1/cube.npz \--set pointing.ra_arcsec=0 --set pointing.dec_arcsec=0 \--set pointing.ra_arcsec=0 --set pointing.dec_arcsec=0 \

    # 設定からガウシアンキューブを内部生成して観測する(pointing が null なら天体中心)
    python run_observe_cube.py --internal --out outputs/observe_demo/observation_internal.npz

対応する入力の条件と、エラーになる条件は fmlo_sim/observation.py の冒頭を参照。
既存の出力ファイルは上書きしない(--overwrite を付けた場合だけ上書き)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fmlo_sim import observation, obsio, pipeline
from fmlo_sim.config import DEFAULT_CONFIG_PATH, load_config, parse_set_args
from fmlo_sim.provenance import collect_provenance
from run_fig6 import _get_git_info

HERE = Path(__file__).resolve().parent


def observe_and_save(out_path, cube_path=None, config_path=None, overrides=None,
                     overwrite=False, cli_argv=None):
    """観測を生成して保存する。(Observation, SkyCube, meta) を返す

    cube_path が None なら設定からガウシアンキューブを内部生成する。
    """
    config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    obsio.refuse_overwrite([out_path], overwrite)   # 計算の前に確認する
    if cube_path is None:
        cfg = load_config(config_path, overrides)
        sky = pipeline.make_internal_sky_cube(cfg)
        setup = pipeline.internal_observing_setup(cfg)
        ignored = ["removal", "evaluation"]
    else:
        sky = observation.load_sky_cube(cube_path)
        setup, ignored = observation.load_observing_setup(config_path, overrides)
    obs = observation.observe(sky, setup)
    provenance = collect_provenance(_get_git_info(), cli_argv, str(config_path))
    provenance["config_overrides"] = overrides or {}
    provenance["config_sections_not_used"] = ignored
    meta = obsio.save_observation(out_path, obs, sky, provenance, overwrite=overwrite)
    return obs, sky, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cube", help="入力キューブ(.npz、cube.save_cube の形式)")
    src.add_argument("--internal", action="store_true", help="設定からガウシアンキューブを内部生成する")
    parser.add_argument("--config", default=None, help="観測設定 YAML(既定: config/default.yaml)")
    parser.add_argument("--out", required=True, help="観測データの出力先(.npz)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="設定の上書き(例: --set pointing.ra_arcsec=20)。複数指定可")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力ファイルを上書きする")
    args = parser.parse_args()

    obs, sky, meta = observe_and_save(
        args.out, cube_path=args.cube, config_path=args.config,
        overrides=parse_set_args(args.set), overwrite=args.overwrite, cli_argv=sys.argv,
    )
    p = meta["pointing"]
    print(f"キューブ: 形 {sky.data.shape} [RA, Dec, 周波数]、{sky.freq_ghz[0]:.6f} - {sky.freq_ghz[-1]:.6f} GHz")
    print(f"指向位置: ({p['requested_ra_arcsec']}, {p['requested_dec_arcsec']}) arcsec → "
          f"画素 {tuple(p['pixel_index'])} = ({p['pixel_ra_arcsec']}, {p['pixel_dec_arcsec']}) arcsec")
    print(f"観測時系列: 形 {obs.observed.shape} [時間, チャンネル]、雑音成分 {meta['noise_components']}")
    if meta["provenance"]["config_sections_not_used"]:
        print(f"使わなかった設定セクション: {meta['provenance']['config_sections_not_used']}")
    print(f"出力: {args.out}(float64・無損失)")


if __name__ == "__main__":
    main()
