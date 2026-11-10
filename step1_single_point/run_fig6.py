"""Taniguchi et al. (2020) Fig. 6 成分推定診断図スクリプト

1点観測の模擬観測データに対し、1回目・2回目の成分推定を実行し、
診断図 (fig09_fig6_diagnostics.png)、復元スペクトル重ね描き
(fig10_fig6_restored_spectra.png) および評価値を出力する。

使い方:
    python run_fig6.py
    python run_fig6.py --config config/default.yaml
    python run_fig6.py --set removal.mean_mode=channel_mean
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from fmlo_sim import evaluation, iterative, pipeline, plotting
from fmlo_sim.config import DEFAULT_CONFIG_PATH, MEAN_MODES, Config, load_config, parse_set_args
from fmlo_sim.modulation import compute_zigzag_period_samples

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_FIG6_CONFIG = HERE / "config" / "fig6_taniguchi2020.yaml"


def _get_git_info() -> dict:
    """gitコミットSHAと未コミット変更の有無を取得。

    失敗時は null と理由を記録する。リポジトリ外やgit未利用でも
    処理を落とさない。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            return {
                "commit_sha": None,
                "commit_sha_reason": f"git rev-parse 失敗 (rc={result.returncode}): {result.stderr.strip()}",
            }
        sha = result.stdout.strip()

        result2 = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        if result2.returncode != 0:
            return {
                "commit_sha": sha,
                "commit_sha_reason": None,
                "uncommitted_modified": None,
                "uncommitted_modified_reason": f"git status 失敗 (rc={result2.returncode}): {result2.stderr.strip()}",
                "uncommitted_untracked": None,
                "uncommitted_untracked_reason": f"git status 失敗 (rc={result2.returncode}): {result2.stderr.strip()}",
            }

        lines = result2.stdout.strip().splitlines() if result2.stdout.strip() else []
        # git status --porcelain のフォーマット: "XY path"
        # X=ステージ状態、Y=ワークツリー状態。両方をチェックし、
        # " M path"（未ステージ変更）も検出する。行頭の空白はstripしない。
        status_chars = {"M", "A", "D", "R", "C", "T", "U"}
        modified = any(
            (len(line) > 0 and line[0] in status_chars) or
            (len(line) > 1 and line[1] in status_chars)
            for line in lines
        )
        # 未追跡ファイルは別フィールド (uncommitted_untracked) に分離し、
        # uncommitted_modified の意味を黙って変更しない。
        untracked = any(line.startswith("??") for line in lines)

        return {
            "commit_sha": sha,
            "commit_sha_reason": None,
            "uncommitted_modified": modified,
            "uncommitted_modified_reason": None,
            "uncommitted_untracked": untracked,
            "uncommitted_untracked_reason": None,
        }
    except FileNotFoundError:
        return {
            "commit_sha": None,
            "commit_sha_reason": "git コマンドが見つかりません",
            "uncommitted_modified": None,
            "uncommitted_modified_reason": "git コマンドが見つかりません",
            "uncommitted_untracked": None,
            "uncommitted_untracked_reason": "git コマンドが見つかりません",
        }
    except Exception as exc:  # noqa: BLE001 - git取得失敗は致命ではない
        return {
            "commit_sha": None,
            "commit_sha_reason": f"予期しないエラー: {exc}",
            "uncommitted_modified": None,
            "uncommitted_modified_reason": f"予期しないエラー: {exc}",
            "uncommitted_untracked": None,
            "uncommitted_untracked_reason": f"予期しないエラー: {exc}",
        }


def _collect_reproducibility_info(
    cfg: Config,
    cli_argv: Optional[list],
    config_path_used: Optional[str],
    shifts_ch: np.ndarray,
) -> dict:
    """実行時の再現性情報を収集する。

    不明な項目は null とその理由を記録する。
    """
    git_info = _get_git_info()
    pattern = cfg.modulation.pattern
    if pattern == "zigzag":
        period_samples, period_seconds = compute_zigzag_period_samples(
            cfg.modulation.half_width_ch,
            cfg.modulation.zigzag_step_ch,
            cfg.modulation.dt_s,
        )
        if period_samples is None:
            period_reason = "定数列で非自明な周期なし"
        else:
            period_reason = None
    elif pattern == "random":
        period_samples, period_seconds, period_reason = (
            None,
            None,
            "randomパターンは周期を持たないため適用外",
        )
    else:
        period_samples, period_seconds, period_reason = (
            None,
            None,
            f"未知の変調パターン: {pattern}",
        )

    return {
        "cli_argv": cli_argv,
        "cli_argv_reason": None if cli_argv is not None else "run()直接呼び出しのため引数列不明",
        "config_path": config_path_used,
        "config_path_reason": None if config_path_used is not None else "設定ファイルパス不明",
        **git_info,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "shifts_ch_min": int(np.min(shifts_ch)) if shifts_ch.size else None,
        "shifts_ch_max": int(np.max(shifts_ch)) if shifts_ch.size else None,
        "configured_half_width_ch": int(cfg.modulation.half_width_ch),
        "effective_half_width_ch": int(max(abs(int(np.min(shifts_ch))), abs(int(np.max(shifts_ch))))) if shifts_ch.size else None,
        "zigzag_period_samples": period_samples,
        "zigzag_period_seconds": period_seconds,
        "zigzag_period_reason": period_reason,
    }


def run(
    cfg: Config,
    outdir: Path | str,
    cutoff_sigma: float = 5.0,
    zoom_half_width_ch: int = 250,
    make_plots: bool = True,
    cli_argv: Optional[list] = None,
    config_path_used: Optional[str] = None,
) -> dict:
    """Fig. 6 の 2 ステップ成分推定を実行する

    Parameters
    ----------
    cfg : シミュレーション設定
    outdir : 出力先ディレクトリ
    cutoff_sigma : sigma-cutoff 閾値 (論文実証解析での典型条件: 5.0)
    zoom_half_width_ch : 拡大診断図の描画半幅 [ch]
    make_plots : 図を保存するかどうか
    cli_argv : コマンドライン引数列 (sys.argv)。None なら null 相当
    config_path_used : 実際に読み込んだ設定ファイルパス。None なら null 相当

    Returns
    -------
    summary : 評価結果および指標の辞書
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 模擬観測データの生成 (1回のみ生成)
    sim = pipeline.simulate(cfg)
    grid = sim.grid

    # 2. 反復計算本体の実行 (独立関数: 観測 TOD・シフト・グリッド情報のみを渡す)
    res = iterative.run_two_step_estimation(
        observed_tod=sim.observed,
        shifts_ch=sim.shifts_ch,
        start0=grid.start0,
        n_wide=grid.n_wide,
        n_components=cfg.removal.n_components,
        mean_mode=cfg.removal.mean_mode,
        cutoff_sigma=cutoff_sigma,
    )

    # 3. 数学的恒等式の検証 (|X - (C_r + A_r + E_r)|)
    err1 = np.max(np.abs(res.x_tod - (res.iter1.c_tod + res.iter1.a_tod + res.iter1.e_tod)))
    err2 = np.max(np.abs(res.x_tod - (res.iter2.c_tod + res.iter2.a_tod + res.iter2.e_tod)))
    u2_err = np.max(np.abs(res.iter2.u_tod - (res.x_tod - res.iter1.a_tod)))

    # 4. 正解データとの客観的評価 (評価時のみ正解を参照)
    m1 = evaluation.evaluate(
        res.iter1.restored_spectrum, sim.truth_k, res.iter1.hits, cfg.n_time,
        grid.freq_ghz, cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
    )
    m2 = evaluation.evaluate(
        res.iter2.restored_spectrum, sim.truth_k, res.iter2.hits, cfg.n_time,
        grid.freq_ghz, cfg.source.center_freq_ghz, cfg.source.line_fwhm_kms,
        cfg.evaluation.line_free_n_fwhm, grid.channel_width_ghz,
    )

    peak_err1 = abs(float(m1["peak_ratio"]) - 1.0)
    peak_err2 = abs(float(m2["peak_ratio"]) - 1.0)
    int_err1 = abs(float(m1["integrated_ratio"]) - 1.0)
    int_err2 = abs(float(m2["integrated_ratio"]) - 1.0)

    summary = {
        "config": cfg.to_dict(),
        "seed": cfg.seed,
        "n_components": cfg.removal.n_components,
        "mean_mode": cfg.removal.mean_mode,
        "cutoff_sigma": float(cutoff_sigma),
        "identities": {
            "max_abs_identity_err_iter1_k": float(err1),
            "max_abs_identity_err_iter2_k": float(err2),
            "max_abs_u2_feedback_err_k": float(u2_err),
            "is_identity_exact": bool(max(err1, err2, u2_err) < 1e-12),
        },
        "first_estimate": {
            **m1,
            "noise_status": res.iter1.noise_status,
            "noise_n_valid": res.iter1.noise_n_valid,
            "explained_ratio": [float(x) for x in res.iter1.explained_ratio],
        },
        "second_estimate": {
            **m2,
            "noise_status": res.iter2.noise_status,
            "noise_n_valid": res.iter2.noise_n_valid,
            "explained_ratio": [float(x) for x in res.iter2.explained_ratio],
        },
        "comparison_iter2_vs_iter1": {
            "peak_ratio_change": float(m2["peak_ratio"] - m1["peak_ratio"]),
            "integrated_ratio_change": float(m2["integrated_ratio"] - m1["integrated_ratio"]),
            "residual_rms_change_k": float(m2["line_free_residual_rms_k"] - m1["line_free_residual_rms_k"]),
            "peak_ratio_abs_error_iter1": peak_err1,
            "peak_ratio_abs_error_iter2": peak_err2,
            "peak_closer_to_truth": bool(peak_err2 < peak_err1),
            "integrated_ratio_abs_error_iter1": int_err1,
            "integrated_ratio_abs_error_iter2": int_err2,
            "integrated_closer_to_truth": bool(int_err2 < int_err1),
        },
        "reproducibility": _collect_reproducibility_info(
            cfg,
            cli_argv,
            config_path_used,
            sim.shifts_ch,
        ),
    }

    # 5. NPZ 保存
    np.savez_compressed(
        outdir / "timestream_fig6.npz",
        time_s=sim.time_s,
        shifts_ch=sim.shifts_ch,
        freq_wide_ghz=grid.freq_ghz,
        start0=grid.start0,
        truth_k=sim.truth_k,
        x_tod=res.x_tod.astype(np.float32),
        # 1st iteration
        u1_tod=res.iter1.u_tod.astype(np.float32),
        c1_tod=res.iter1.c_tod.astype(np.float32),
        a1_tod=res.iter1.a_tod.astype(np.float32),
        e1_tod=res.iter1.e_tod.astype(np.float32),
        restored1_k=res.iter1.restored_spectrum.astype(np.float32),
        model1_k=res.iter1.model_spectrum.astype(np.float32),
        noise_sigma1_k=res.iter1.noise_sigma.astype(np.float32),
        # 2nd iteration
        u2_tod=res.iter2.u_tod.astype(np.float32),
        c2_tod=res.iter2.c_tod.astype(np.float32),
        a2_tod=res.iter2.a_tod.astype(np.float32),
        e2_tod=res.iter2.e_tod.astype(np.float32),
        restored2_k=res.iter2.restored_spectrum.astype(np.float32),
        model2_k=res.iter2.model_spectrum.astype(np.float32),
        noise_sigma2_k=res.iter2.noise_sigma.astype(np.float32),
        hits=res.iter1.hits,
    )

    # 6. JSON 保存
    with open(outdir / "metrics_fig6.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 7. プロット作成
    if make_plots:
        # (a) 全チャンネル版
        plotting.plot_fig6_diagnostics(
            outdir / "fig09_fig6_diagnostics_full.png",
            res,
            sim.time_s,
            ch_range=None,
            title_suffix=" (Full 4096 ch)",
        )

        # (b) 輝線中心付近の拡大版 (既定: center +- zoom_half_width_ch)
        center_ch = grid.n_channels // 2
        zoom_range = (center_ch - zoom_half_width_ch, center_ch + zoom_half_width_ch)
        plotting.plot_fig6_diagnostics(
            outdir / "fig09_fig6_diagnostics.png",
            res,
            sim.time_s,
            ch_range=zoom_range,
            title_suffix=f" (Zoom: ch [{zoom_range[0]}, {zoom_range[1]}])",
        )

        # (c) 復元スペクトルの重ね描き (B-1)
        plotting.plot_fig6_restored_spectra(
            outdir / "fig10_fig6_restored_spectra.png",
            grid.freq_ghz,
            sim.truth_k,
            res.iter1.restored_spectrum,
            res.iter2.restored_spectrum,
            res.iter1.hits,
            len(sim.time_s),
            cfg.source.center_freq_ghz,
            cfg.source.line_fwhm_kms,
        )

    return summary


def print_summary(summary: dict) -> None:
    print("=" * 60)
    print("Taniguchi et al. (2020) Fig. 6 2ステップ成分推定サマリー")
    print("=" * 60)
    id_info = summary["identities"]
    print(f"恒等式 X = C_r + A_r + E_r 最大絶対誤差: 1回目={id_info['max_abs_identity_err_iter1_k']:.2e} K, "
          f"2回目={id_info['max_abs_identity_err_iter2_k']:.2e} K (厳密成立: {id_info['is_identity_exact']})")
    print(f"フィードバック U_2 = X - A_1 最大絶対誤差: {id_info['max_abs_u2_feedback_err_k']:.2e} K")
    print("-" * 60)
    print(f"{'Estimate':<16}{'ピーク比':>10}{'積分強度比':>12}{'残差RMS [K]':>14}")
    m1 = summary["first_estimate"]
    m2 = summary["second_estimate"]
    print(f"{'1st estimate':<16}{m1['peak_ratio']:>10.4f}{m1['integrated_ratio']:>12.4f}"
          f"{m1['line_free_residual_rms_k']:>14.4f}")
    print(f"{'2nd estimate':<16}{m2['peak_ratio']:>10.4f}{m2['integrated_ratio']:>12.4f}"
          f"{m2['line_free_residual_rms_k']:>14.4f}")
    print("-" * 60)
    cmp = summary["comparison_iter2_vs_iter1"]
    print(f"2回目の変化: ピーク比 {cmp['peak_ratio_change']:+.4f}, "
          f"積分強度比 {cmp['integrated_ratio_change']:+.4f}, "
          f"残差RMS {cmp['residual_rms_change_k']:+.4e} K")
    p_status = "正解に接近" if cmp["peak_closer_to_truth"] else "正解から乖離または不変"
    i_status = "正解に接近" if cmp["integrated_closer_to_truth"] else "正解から乖離または不変"
    print(f"ピーク比誤差: |r-1| {cmp['peak_ratio_abs_error_iter1']:.4f} -> {cmp['peak_ratio_abs_error_iter2']:.4f} ({p_status})")
    print(f"積分強度比誤差: |r-1| {cmp['integrated_ratio_abs_error_iter1']:.4f} -> {cmp['integrated_ratio_abs_error_iter2']:.4f} ({i_status})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=None,
        help="設定ファイル (既定: config/fig6_taniguchi2020.yaml があれば優先、なければ config/default.yaml)"
    )
    parser.add_argument(
        "--outdir", default="outputs/fig6",
        help="出力ディレクトリ (既定: outputs/fig6)"
    )
    parser.add_argument(
        "--cutoff-sigma", type=float, default=5.0,
        help="sigma-cutoff の閾値 theta_cutoff (既定: 5.0)"
    )
    parser.add_argument(
        "--zoom-half-width-ch", type=int, default=250,
        help="診断図の拡大表示の半幅 [ch] (既定: 250)"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="設定の上書き (例: --set removal.mean_mode=channel_mean)"
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        if DEFAULT_FIG6_CONFIG.is_file():
            config_path = DEFAULT_FIG6_CONFIG
        else:
            config_path = DEFAULT_CONFIG_PATH

    overrides = parse_set_args(args.overrides)
    cfg = load_config(config_path, overrides)

    outdir = HERE / args.outdir
    summary = run(
        cfg,
        outdir,
        cutoff_sigma=args.cutoff_sigma,
        zoom_half_width_ch=args.zoom_half_width_ch,
        cli_argv=sys.argv,
        config_path_used=str(config_path),
    )
    print_summary(summary)
    print(f"出力先: {outdir.resolve()}")


if __name__ == "__main__":
    main()