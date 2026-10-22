"""単体テスト: PCA および FA+ICA 雑音除去の数学的性質・形状・不変性の検証"""
import numpy as np
import pytest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fmlo_sim import removal


@pytest.fixture
def dummy_tod():
    """テスト用の模擬データ (n_time=100, n_channels=256)"""
    rng = np.random.default_rng(12345)
    return rng.normal(loc=5.0, scale=2.0, size=(100, 256))


def test_pca_shapes_and_invariance(dummy_tod):
    """PCA: 形状と cleaned + removed == tod の確認"""
    orig = dummy_tod.copy()
    res = removal.pca_remove(dummy_tod, n_components=2, mean_mode="channel_mean")

    assert np.array_equal(dummy_tod, orig), "入力配列が in-place で変更されています"
    assert res.cleaned.shape == dummy_tod.shape
    assert res.removed.shape == dummy_tod.shape
    assert np.allclose(res.cleaned + res.removed, dummy_tod, atol=1e-12)


def test_fa_ica_shapes_and_invariance(dummy_tod):
    """FA+ICA: 形状と cleaned + removed == tod の確認 (両モード)"""
    orig = dummy_tod.copy()

    for mode in ("channel_mean", "none"):
        res = removal.fa_ica_remove(dummy_tod, n_factors=3, remove_indices=[0, 1], mean_mode=mode)
        assert np.array_equal(dummy_tod, orig), f"入力配列が in-place で変更されています (mode={mode})"
        assert res.cleaned.shape == dummy_tod.shape
        assert res.removed.shape == dummy_tod.shape
        assert np.allclose(res.cleaned + res.removed, dummy_tod, atol=1e-12), f"再構成誤差があります (mode={mode})"
        assert res.n_factors == 3
        assert res.remove_indices == [0, 1]


def test_fa_ica_all_components_equals_fa_common(dummy_tod):
    """全ICA成分を除去した結果が、FA共通成分 (Z @ L) と数学的に厳密一致するか検証"""
    res = removal.fa_ica_remove(dummy_tod, n_factors=3, remove_indices=None, mean_mode="channel_mean")
    
    # S @ mixing.T @ loadings vs Z @ loadings
    recon_ica_all = res.ica_components @ res.effective_profiles
    recon_fa = res.fa_scores @ res.fa_loadings
    assert np.allclose(recon_ica_all, recon_fa, atol=1e-12), "全ICA成分再構成がFA共通成分と一致しません"


def test_fa_ica_empty_removal(dummy_tod):
    """除去成分が空リストの場合、channel_mean では平均のみが除去され、none では恒等変換になる"""
    res_cm = removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[], mean_mode="channel_mean")
    mean = dummy_tod.mean(axis=0)
    assert np.allclose(res_cm.removed, mean, atol=1e-12)

    res_none = removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[], mean_mode="none")
    assert np.allclose(res_none.cleaned, dummy_tod, atol=1e-12)
    assert np.allclose(res_none.removed, 0.0, atol=1e-12)


def test_fa_ica_reproducibility(dummy_tod):
    """seed 固定で完全に再現することを確認"""
    res1 = removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[0], random_state=42)
    res2 = removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[0], random_state=42)

    assert np.array_equal(res1.cleaned, res2.cleaned)
    assert np.array_equal(res1.removed, res2.removed)


def test_fa_ica_invalid_inputs(dummy_tod):
    """不正な引数のエラー検出"""
    # 不正な n_factors
    with pytest.raises(ValueError):
        removal.fa_ica_remove(dummy_tod, n_factors=0)
    with pytest.raises(ValueError):
        removal.fa_ica_remove(dummy_tod, n_factors=-1)

    # 不正な remove_indices
    with pytest.raises(ValueError):
        removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[2])  # 2 は範囲外 (0, 1)
    with pytest.raises(ValueError):
        removal.fa_ica_remove(dummy_tod, n_factors=2, remove_indices=[-1])

    # 不正な mean_mode
    with pytest.raises(ValueError):
        removal.fa_ica_remove(dummy_tod, n_factors=2, mean_mode="invalid_mode")
