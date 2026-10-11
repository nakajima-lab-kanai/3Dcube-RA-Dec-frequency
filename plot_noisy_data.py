import argparse
import numpy as np
import matplotlib.pyplot as plt

# nobeyama_fmlo_simulation から関数をインポート
from nobeyama_fmlo_simulation import (
    generate_scan_path, 
    generate_sky_model, 
    apply_fmlo_modulation, 
    generate_shift_pattern,
    generate_noise, 
    fmlo_demodulate, 
    grid_data
)

def plot_noisy_data(modulation_width=15, modulation_step=3, noise_type='laminar', output_path='noisy_data_visualization.png'):
    print(f"ノイズ添加データのシミュレーションを開始します (変調幅: ±{modulation_width}ch, ステップ: {modulation_step}ch, ノイズ: {noise_type})...")
    # パラメータ（元コードと同一）
    extent = [-150, 150, -150, 150]
    n_lines = 30
    points_per_line = 100
    n_time = n_lines * points_per_line
    n_channels = 256
    center_ch = 128
    
    # 軌跡とシグナルの生成
    ra_x, dec_x = generate_scan_path('x', n_lines, points_per_line, extent)
    clean_signal_x = generate_sky_model(ra_x, dec_x, n_channels, center_ch)
    
    # FMLO変調
    shift_pattern = generate_shift_pattern(n_time, modulation_width=modulation_width, modulation_step=modulation_step)
    mod_signal_x = apply_fmlo_modulation(clean_signal_x, shift_pattern)
    
    # ノイズ付加（相関ノイズ＋熱雑音）
    noise = generate_noise(n_time, n_channels, noise_type=noise_type)
    obs_x = mod_signal_x + noise
    
    # 描画
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. タイムストリーム（ウォーターフォールプロット）の可視化
    # 変調の様子や相関ノイズの構造を見えやすくするため、最初の500ステップだけを描画
    time_subset = slice(0, 500)
    im1 = axs[0].imshow(obs_x[time_subset, :].T, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title(f'Noisy Timestream (First 500 steps)\n[Mod width: ±{modulation_width}ch, step: {modulation_step}ch, Noise: {noise_type}]')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Frequency Channel')
    fig.colorbar(im1, ax=axs[0], label='Intensity')
    
    # 2. ダーティマップ（PCAなし）の可視化
    # 復調は行うが、PCAによるベースライン除去を行わずにマップ化した場合どうなるか
    demod_x = fmlo_demodulate(obs_x, shift_pattern)
    integ_range = 15
    integ_x = np.sum(demod_x[:, center_ch-integ_range:center_ch+integ_range], axis=1)
    
    _, _, map_x = grid_data(ra_x, dec_x, integ_x, extent)
    
    im2 = axs[1].imshow(map_x.T, origin='lower', extent=extent, cmap='magma')
    axs[1].set_title(f'Dirty Map (x-scan, No PCA Baseline Removal)\n[Noise: {noise_type}]')
    axs[1].set_xlabel('Relative RA (arcsec)')
    axs[1].set_ylabel('Relative Dec (arcsec)')
    fig.colorbar(im2, ax=axs[1], label='Integrated Intensity')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"完了しました。可視化画像を '{output_path}' に保存しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ノイズ添加データの可視化スクリプト")
    parser.add_argument("--mod-width", "--modulation-width", dest="mod_width", type=int, default=15,
                        help="変調幅（最大シフト量 [ch]、既定値: 15）")
    parser.add_argument("--mod-step", "--modulation-step", dest="mod_step", type=int, default=3,
                        help="変調ステップ（1ステップあたりのシフト量 [ch]、既定値: 3）")
    parser.add_argument("--noise-type", choices=['laminar', 'turbulent'], default='laminar',
                        help="大気雑音モデル ('laminar': 層流, 'turbulent': 乱流、既定値: laminar)")
    parser.add_argument("--output", default='noisy_data_visualization.png',
                        help="出力画像ファイルパス (既定値: noisy_data_visualization.png)")
    args = parser.parse_args()
    plot_noisy_data(modulation_width=args.mod_width, modulation_step=args.mod_step,
                    noise_type=args.noise_type, output_path=args.output)
