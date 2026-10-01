import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

def generate_scan_path(scan_type, n_lines, points_per_line, extent):
    """
    スキャン軌跡を生成する
    extent: [ra_min, ra_max, dec_min, dec_max]
    """
    ra_min, ra_max, dec_min, dec_max = extent
    ra = np.zeros(n_lines * points_per_line)
    dec = np.zeros(n_lines * points_per_line)
    
    for i in range(n_lines):
        start_idx = i * points_per_line
        end_idx = (i + 1) * points_per_line
        if scan_type == 'x':
            # RA方向にスキャン
            dec[start_idx:end_idx] = dec_min + (dec_max - dec_min) * i / (n_lines - 1)
            ra_line = np.linspace(ra_min, ra_max, points_per_line)
            if i % 2 == 1:
                ra_line = ra_line[::-1] # ジグザグ
            ra[start_idx:end_idx] = ra_line
        elif scan_type == 'y':
            # Dec方向にスキャン
            ra[start_idx:end_idx] = ra_min + (ra_max - ra_min) * i / (n_lines - 1)
            dec_line = np.linspace(dec_min, dec_max, points_per_line)
            if i % 2 == 1:
                dec_line = dec_line[::-1]
            dec[start_idx:end_idx] = dec_line
            
    return ra, dec

def generate_sky_model(ra, dec, n_channels, center_ch, source_ra=0, source_dec=0, size=0.5):
    """
    天体の輝線シグナルを生成
    """
    signal = np.zeros((len(ra), n_channels))
    # 2次元ガウシアンの空間分布
    spatial = np.exp(-((ra - source_ra)**2 + (dec - source_dec)**2) / (2 * size**2))
    
    # ガウシアンのスペクトル分布
    ch_array = np.arange(n_channels)
    spectral = np.exp(-((ch_array - center_ch)**2) / (2 * 2**2)) # line width = 2 channels
    
    for t in range(len(ra)):
        signal[t, :] = spatial[t] * spectral
        
    return signal * 10.0 # 信号強度

def apply_fmlo_modulation(signal, shift_pattern):
    """
    FMLO変調 (周波数をシフト)
    """
    n_time, n_channels = signal.shape
    modulated = np.zeros_like(signal)
    for t in range(n_time):
        shift = shift_pattern[t]
        # np.rollで周波数シフト（循環シフト）
        modulated[t, :] = np.roll(signal[t, :], shift)
    return modulated

def generate_correlated_noise(n_time, n_channels):
    """
    大気や装置由来の時間変動する相関ノイズ (1/f ノイズのような挙動を模擬)
    """
    # チャンネル間で相関のあるベースライン
    # 時間変化するスロープやオフセット
    t_array = np.linspace(0, 10, n_time)
    ch_array = np.linspace(-1, 1, n_channels)
    
    noise = np.zeros((n_time, n_channels))
    # ゆっくりとした時間変動
    time_var1 = np.sin(2 * np.pi * 0.1 * t_array) + 2.0
    time_var2 = np.cos(2 * np.pi * 0.05 * t_array)
    
    for t in range(n_time):
        # チャンネル方向に相関を持つパターン (例えば傾きや二次関数)
        noise[t, :] = time_var1[t] * 100.0 + time_var2[t] * 20.0 * ch_array + np.random.normal(0, 5, n_channels)
        
    return noise

def fmlo_demodulate(data, shift_pattern):
    """
    FMLO復調 (逆シフト)
    """
    n_time, n_channels = data.shape
    demodulated = np.zeros_like(data)
    for t in range(n_time):
        shift = shift_pattern[t]
        demodulated[t, :] = np.roll(data[t, :], -shift)
    return demodulated

def grid_data(ra, dec, values, extent, grid_size=50):
    """
    タイムストリームの値を2次元グリッドにマッピングする
    """
    ra_min, ra_max, dec_min, dec_max = extent
    grid_ra, grid_dec = np.mgrid[ra_min:ra_max:complex(0, grid_size), dec_min:dec_max:complex(0, grid_size)]
    grid_z = griddata((ra, dec), values, (grid_ra, grid_dec), method='linear')
    return grid_ra, grid_dec, grid_z

def main():
    # パラメータ設定
    extent = [-2, 2, -2, 2] # RA, Decの範囲 (相対座標)
    n_lines = 20
    points_per_line = 100
    n_time = n_lines * points_per_line
    n_channels = 128
    center_ch = 64
    
    # 1. スキャン軌跡の生成 (x-scan, y-scan)
    ra_x, dec_x = generate_scan_path('x', n_lines, points_per_line, extent)
    ra_y, dec_y = generate_scan_path('y', n_lines, points_per_line, extent)
    
    # 2. 天体シグナルの生成
    signal_x = generate_sky_model(ra_x, dec_x, n_channels, center_ch)
    signal_y = generate_sky_model(ra_y, dec_y, n_channels, center_ch)
    
    # 3. FMLO変調パターンの生成 (ジグザグ)
    # -10 から 10 チャンネルのシフトを繰り返す
    modulation_period = 10
    shift_pattern = (np.arange(n_time) % modulation_period) - (modulation_period // 2)
    shift_pattern = shift_pattern * 2 # 振幅
    
    mod_signal_x = apply_fmlo_modulation(signal_x, shift_pattern)
    mod_signal_y = apply_fmlo_modulation(signal_y, shift_pattern)
    
    # 4. 相関ノイズ(大気など)の付加
    noise_x = generate_correlated_noise(n_time, n_channels)
    noise_y = generate_correlated_noise(n_time, n_channels)
    
    obs_x = mod_signal_x + noise_x
    obs_y = mod_signal_y + noise_y
    
    # 5. PCAを用いたデータ解析 (相関ノイズの除去)
    pca = PCA(n_components=2) # 上位2成分をノイズとして扱う
    
    # x-scanの処理
    pca.fit(obs_x)
    baseline_x = pca.inverse_transform(pca.transform(obs_x))
    cleaned_x = obs_x - baseline_x
    
    # y-scanの処理
    pca.fit(obs_y)
    baseline_y = pca.inverse_transform(pca.transform(obs_y))
    cleaned_y = obs_y - baseline_y
    
    # 6. FMLO復調
    demod_x = fmlo_demodulate(cleaned_x, shift_pattern)
    demod_y = fmlo_demodulate(cleaned_y, shift_pattern)
    
    # 7. 積分強度の計算 (中心チャンネル周辺を足し合わせる)
    integ_x = np.sum(demod_x[:, center_ch-5:center_ch+5], axis=1)
    integ_y = np.sum(demod_y[:, center_ch-5:center_ch+5], axis=1)
    
    # 8. マップの生成 (グリッディング)
    grid_ra, grid_dec, map_x = grid_data(ra_x, dec_x, integ_x, extent)
    _, _, map_y = grid_data(ra_y, dec_y, integ_y, extent)
    
    # 9. Basket Weave (単純な平均による統合)
    # スキャンノイズ(ストライプ)が直交しているため、平均することで相殺効果がある
    basket_weave_map = (np.nan_to_num(map_x) + np.nan_to_num(map_y)) / 2.0
    
    # 10. 可視化
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axs[0].imshow(map_x.T, origin='lower', extent=extent, cmap='viridis', vmin=-10, vmax=50)
    axs[0].set_title('x-scan Map')
    axs[0].set_xlabel('Relative RA')
    axs[0].set_ylabel('Relative Dec')
    fig.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(map_y.T, origin='lower', extent=extent, cmap='viridis', vmin=-10, vmax=50)
    axs[1].set_title('y-scan Map')
    axs[1].set_xlabel('Relative RA')
    axs[1].set_ylabel('Relative Dec')
    fig.colorbar(im2, ax=axs[1])
    
    im3 = axs[2].imshow(basket_weave_map.T, origin='lower', extent=extent, cmap='viridis', vmin=-10, vmax=50)
    axs[2].set_title('Basket Weave Map (Combined)')
    axs[2].set_xlabel('Relative RA')
    axs[2].set_ylabel('Relative Dec')
    fig.colorbar(im3, ax=axs[2])
    
    plt.tight_layout()
    plt.savefig('fmlo_basket_weave_results.png')
    print("Simulation complete. Results saved to 'fmlo_basket_weave_results.png'")

if __name__ == '__main__':
    main()
