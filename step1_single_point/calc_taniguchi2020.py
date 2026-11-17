import numpy as np
from fmlo_sim.taniguchi2020 import load_fig6_config, run_fig6

cfg = load_fig6_config()
truth, res = run_fig6(cfg)

# 輝線マスク（真の輝線が 0.01 を超えるチャンネルを輝線領域とみなす）
line_mask = truth.line_spectrum > 0.01
bg_mask = (~line_mask) & (res.hits > 50)  # サンプル数が十分なベースライン領域

for i, st in enumerate(res.steps, 1):
    # 1. 相関成分の相対誤差（対真値）
    cor_rel_err = np.linalg.norm(st.t_cor - truth.t_cor) / np.linalg.norm(truth.t_cor)
    
    # 2. 残差 E^nc の統計量（理論値: mean=0.0, std=1.0）
    enc_mean, enc_std = st.e_nc.mean(), st.e_nc.std()
    
    # 3. ノイズ寄与係数 alpha_FMLO（論文基準: ~1.10）
    ideal_noise = cfg.noise.sigma / np.sqrt(res.hits[bg_mask])
    alpha_fmlo = np.std(st.spectrum[bg_mask] / ideal_noise)
    
    # 4. 輝線のピーク強度比・積分強度比（対真値）
    peak_ratio = st.spectrum[line_mask].max() / truth.line_spectrum[line_mask].max()
    flux_ratio = st.spectrum[line_mask].sum() / truth.line_spectrum[line_mask].sum()
    
    print(f"=== Iteration {i} (K={st.n_components}) ===")
    print(f"  Variation (Eq.38)   : {st.variation if st.variation is not None else 'N/A'}")
    print(f"  T^cor Rel. Error    : {cor_rel_err:.4f}")
    print(f"  E^nc Mean / Std     : {enc_mean:+.4f} / {enc_std:.4f}")
    print(f"  alpha_FMLO (Eq.42)  : {alpha_fmlo:.4f}")
    print(f"  Line Peak / Flux    : {peak_ratio:.4f} / {flux_ratio:.4f}")