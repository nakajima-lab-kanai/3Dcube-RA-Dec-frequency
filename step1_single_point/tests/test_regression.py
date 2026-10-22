"""回帰テスト: FA+ICA 追加後も既存 PCA の数値が完全一致することを検証する"""
import pytest
import numpy as np
from pathlib import Path
import sys

# step1_single_point をパスに追加
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fmlo_sim import config, pipeline


def test_baseline_pca_regression():
    """既定設定 (seed=20261005, offset_k=0.0) での PCA 数値の一致確認"""
    cfg = config.load_config()
    sim = pipeline.simulate(cfg)
    results = pipeline.reduce_all_modes(sim)

    # 変更前ベースラインの記録値
    expected = {
        "none": {
            "line_free_residual_rms_k": 0.039404,
            "peak_ratio": 0.987595,
            "integrated_ratio": 0.933568,
        },
        "channel_mean": {
            "line_free_residual_rms_k": 0.042941,
            "peak_ratio": 0.830907,
            "integrated_ratio": 0.432657,
        },
        "no_pca": {
            "line_free_residual_rms_k": 0.037233,
            "peak_ratio": 1.000494,
            "integrated_ratio": 0.983794,
        },
    }

    for label, exp_metrics in expected.items():
        assert label in results, f"Result label '{label}' not found"
        act_metrics = results[label].metrics
        for k, v in exp_metrics.items():
            assert np.isclose(act_metrics[k], v, atol=1e-4), (
                f"Mismatch in {label} {k}: actual={act_metrics[k]}, expected={v}"
            )
