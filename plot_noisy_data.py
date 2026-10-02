import numpy as np
import matplotlib.pyplot as plt

# nobeyama_fmlo_simulation から関数をインポート
from nobeyama_fmlo_simulation import (
    generate_scan_path, 
    generate_sky_model, 
    apply_fmlo_modulation, 
    generate_noise, 
    fmlo_demodulate, 
    grid_data
)

def plot_noisy_data():
    print("ノイズ添加データのシミュレーションを開始します...")
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
    modulation_period = 20
    shift_pattern = np.abs((np.arange(n_time) % modulation_period) - (modulation_period / 2)) - (modulation_period / 4)
    shift_pattern = shift_pattern * 3.0
    mod_signal_x = apply_fmlo_modulation(clean_signal_x, shift_pattern)
    
    # ノイズ付加（相関ノイズ＋熱雑音）
    noise = generate_noise(n_time, n_channels)
    obs_x = mod_signal_x + noise
    
    # 描画
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. タイムストリーム（ウォーターフォールプロット）の可視化
    # 変調の様子や相関ノイズの構造を見えやすくするため、最初の500ステップだけを描画
    time_subset = slice(0, 500)
    im1 = axs[0].imshow(obs_x[time_subset, :].T, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title('Noisy Timestream (First 500 steps)\nTime vs Frequency Channels')
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
    axs[1].set_title('Dirty Map (x-scan, No PCA Baseline Removal)')
    axs[1].set_xlabel('Relative RA (arcsec)')
    axs[1].set_ylabel('Relative Dec (arcsec)')
    fig.colorbar(im2, ax=axs[1], label='Integrated Intensity')
    
    plt.tight_layout()
    plt.savefig('noisy_data_visualization.png', dpi=150)
    print("完了しました。可視化画像を 'noisy_data_visualization.png' に保存しました。")

if __name__ == '__main__':
    plot_noisy_data()
