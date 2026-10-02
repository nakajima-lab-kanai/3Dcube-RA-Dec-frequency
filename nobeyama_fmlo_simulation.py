import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

def generate_scan_path(scan_type, n_lines, points_per_line, extent):
    """
    スキャン軌跡を生成する関数
    
    Parameters:
    - scan_type: 'x' (RA方向のラスタースキャン) または 'y' (Dec方向のラスタースキャン)
    - n_lines: スキャンの行（列）数
    - points_per_line: 1行あたりの観測点（タイムステップ）数
    - extent: [ra_min, ra_max, dec_min, dec_max] (単位: arcsec)
    
    Returns:
    - ra, dec: 時間経過に伴うRA, Decの1次元配列（タイムストリームの座標）
    """
    ra_min, ra_max, dec_min, dec_max = extent
    ra = np.zeros(n_lines * points_per_line)
    dec = np.zeros(n_lines * points_per_line)
    
    for i in range(n_lines):
        start_idx = i * points_per_line
        end_idx = (i + 1) * points_per_line
        if scan_type == 'x':
            # RA方向に往復スキャン（ジグザグ）
            dec_pos = dec_min + (dec_max - dec_min) * i / (max(n_lines - 1, 1))
            dec[start_idx:end_idx] = dec_pos
            ra_line = np.linspace(ra_min, ra_max, points_per_line)
            if i % 2 == 1:
                ra_line = ra_line[::-1] # 偶数行は逆方向
            ra[start_idx:end_idx] = ra_line
        elif scan_type == 'y':
            # Dec方向に往復スキャン
            ra_pos = ra_min + (ra_max - ra_min) * i / (max(n_lines - 1, 1))
            ra[start_idx:end_idx] = ra_pos
            dec_line = np.linspace(dec_min, dec_max, points_per_line)
            if i % 2 == 1:
                dec_line = dec_line[::-1]
            dec[start_idx:end_idx] = dec_line
            
    return ra, dec

def generate_sky_model(ra, dec, n_channels, center_ch, spatial_size=50.0, velocity_width_ch=25.0):
    """
    天体の輝線シグナルを生成する関数（野辺山45m 150GHz帯を想定したガウシアンボール）
    
    Parameters:
    - ra, dec: 観測座標のタイムストリーム (arcsec)
    - n_channels: 分光計のチャンネル数
    - center_ch: 輝線の中心チャンネル
    - spatial_size: 天体の空間サイズ (arcsec) - 例: 50 arcsec
    - velocity_width_ch: 輝線の速度幅（チャンネル換算） - 例: 150GHz帯で50km/sなら約25MHz幅。1ch=1MHzと仮定して25ch。
    
    Returns:
    - signal: (n_time, n_channels) のタイムストリームデータ
    """
    n_time = len(ra)
    signal = np.zeros((n_time, n_channels))
    
    # 2次元ガウシアンによる空間分布 (原点中心)
    spatial_profile = np.exp(-((ra)**2 + (dec)**2) / (2 * (spatial_size / 2.355)**2)) # FWHMからsigmaへ換算
    
    # 1次元ガウシアンによるスペクトル分布（輝線）
    ch_array = np.arange(n_channels)
    # FWHM = 25チャンネルとした場合のsigma
    sigma_ch = velocity_width_ch / 2.355
    spectral_profile = np.exp(-((ch_array - center_ch)**2) / (2 * sigma_ch**2))
    
    # 空間×スペクトルで時間ごとのシグナルを計算
    for t in range(n_time):
        signal[t, :] = spatial_profile[t] * spectral_profile
        
    return signal * 5.0 # 天体信号のピーク強度（任意単位）

def apply_fmlo_modulation(signal, shift_pattern):
    """
    FMLO（周波数変調局部発振器）法による変調処理
    観測中にLO周波数を高速に変化させることで、天体シグナルの受信周波数を意図的にシフトさせます。
    これにより、天体シグナルは時間に対して「非相関」な変動を持つようになります。
    """
    n_time, n_channels = signal.shape
    modulated = np.zeros_like(signal)
    for t in range(n_time):
        shift = int(shift_pattern[t])
        # np.rollを用いて周波数チャンネルをシフト（はみ出した部分は循環シフト）
        modulated[t, :] = np.roll(signal[t, :], shift)
    return modulated

def fmlo_demodulate(data, shift_pattern):
    """
    FMLO復調処理（逆シフト）
    データ解析段階で、変調時と逆のシフト量を適用します。
    これにより、天体シグナルの周波数位置が本来のチャンネルに揃えられ、積分可能になります。
    """
    n_time, n_channels = data.shape
    demodulated = np.zeros_like(data)
    for t in range(n_time):
        shift = int(shift_pattern[t])
        demodulated[t, :] = np.roll(data[t, :], -shift)
    return demodulated

def generate_noise(n_time, n_channels):
    """
    大気や装置に由来する相関ノイズと、熱雑音（ホワイトノイズ）を生成する
    """
    # 1. 相関ノイズ（大気放射や装置のベースライン変動）
    # 低周波成分が強い（1/fノイズ的）挙動を簡単なサイン波の合成で模擬
    t_array = np.linspace(0, 5, n_time)
    ch_array = np.linspace(-1, 1, n_channels)
    
    correlated_noise = np.zeros((n_time, n_channels))
    time_fluctuation = 20.0 * np.sin(2 * np.pi * 0.5 * t_array) + 10.0 * np.cos(2 * np.pi * 1.5 * t_array) + 50.0
    
    for t in range(n_time):
        # チャンネル方向にも傾きやうねりを持たせる
        correlated_noise[t, :] = time_fluctuation[t] + 5.0 * np.sin(2 * np.pi * 1.0 * ch_array) * t_array[t]
        
    # 2. 熱雑音（ホワイトノイズ）
    thermal_noise = np.random.normal(0, 2.0, (n_time, n_channels))
    
    return correlated_noise + thermal_noise

def grid_data(ra, dec, values, extent, grid_size=60):
    """
    タイムストリームの値を2次元の積分強度マップ（画像）に再構成（グリッディング）する
    """
    ra_min, ra_max, dec_min, dec_max = extent
    grid_ra, grid_dec = np.mgrid[ra_min:ra_max:complex(0, grid_size), dec_min:dec_max:complex(0, grid_size)]
    grid_z = griddata((ra, dec), values, (grid_ra, grid_dec), method='cubic', fill_value=0)
    return grid_ra, grid_dec, grid_z

def main():
    # --- 観測パラメータ設定 ---
    extent = [-150, 150, -150, 150] # マップ範囲 (arcsec)
    n_lines = 30                    # スキャン本数
    points_per_line = 100           # 1スキャンあたりのデータ点数
    n_time = n_lines * points_per_line
    n_channels = 256                # 分光チャンネル数
    center_ch = 128                 # 輝線の中心チャンネル
    
    print("1. スキャン軌跡の生成...")
    ra_x, dec_x = generate_scan_path('x', n_lines, points_per_line, extent)
    ra_y, dec_y = generate_scan_path('y', n_lines, points_per_line, extent)
    
    print("2. 模擬天体シグナル（50 arcsec, 50 km/s幅想定）の生成...")
    clean_signal_x = generate_sky_model(ra_x, dec_x, n_channels, center_ch)
    clean_signal_y = generate_sky_model(ra_y, dec_y, n_channels, center_ch)
    
    print("3. FMLO変調の適用...")
    # 変調パターン：例えば観測中にチャンネルを±15シフトさせる三角波（ジグザグ）パターン
    modulation_period = 20 # 20タイムステップで1周期
    shift_pattern = np.abs((np.arange(n_time) % modulation_period) - (modulation_period / 2)) - (modulation_period / 4)
    shift_pattern = shift_pattern * 3.0 # 振幅調整
    
    mod_signal_x = apply_fmlo_modulation(clean_signal_x, shift_pattern)
    mod_signal_y = apply_fmlo_modulation(clean_signal_y, shift_pattern)
    
    print("4. 大気相関ノイズおよび熱雑音の付加...")
    obs_x = mod_signal_x + generate_noise(n_time, n_channels)
    obs_y = mod_signal_y + generate_noise(n_time, n_channels)
    
    print("5. データ解析（DE:MIST / PCAによる相関ノイズ除去）...")
    # 主成分分析(PCA)を用いて、時間・チャンネル間で強い相関を持つ成分（大気ノイズなど）を抽出
    # FMLOで変調された天体シグナルは相関が崩れているため、PCAの上位成分には含まれにくくなります
    pca = PCA(n_components=3) 
    
    # x-scanのノイズ除去
    pca.fit(obs_x)
    baseline_x = pca.inverse_transform(pca.transform(obs_x))
    cleaned_x = obs_x - baseline_x # ベースライン成分を引き去る
    
    # y-scanのノイズ除去
    pca.fit(obs_y)
    baseline_y = pca.inverse_transform(pca.transform(obs_y))
    cleaned_y = obs_y - baseline_y
    
    print("6. 信号の復調...")
    demod_x = fmlo_demodulate(cleaned_x, shift_pattern)
    demod_y = fmlo_demodulate(cleaned_y, shift_pattern)
    
    print("7. 積分強度の計算とマップ化...")
    # 中心チャンネル周辺（輝線が存在する帯域）を足し合わせて積分強度を算出
    integ_range = 15 # ±15チャンネルを積分
    integ_x = np.sum(demod_x[:, center_ch-integ_range:center_ch+integ_range], axis=1)
    integ_y = np.sum(demod_y[:, center_ch-integ_range:center_ch+integ_range], axis=1)
    
    _, _, map_x = grid_data(ra_x, dec_x, integ_x, extent)
    _, _, map_y = grid_data(ra_y, dec_y, integ_y, extent)
    
    print("8. Basket Weave処理...")
    # x-scanとy-scanを統合。単純な平均のほか、フーリエ変換等を用いたストライプ除去アルゴリズムが
    # 本格的なBasket Weaveですが、ここではシミュレーションとして重み付けなしの平均を採用し
    # 直交するスキャンノイズを効果的に相殺します。
    basket_map = (np.nan_to_num(map_x) + np.nan_to_num(map_y)) / 2.0
    
    print("9. 描画と保存...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 共通のカラースケール
    vmin, vmax = np.nanmin(basket_map), np.nanmax(basket_map)
    
    im1 = axs[0].imshow(map_x.T, origin='lower', extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axs[0].set_title('x-scan Map (RA Scanning)')
    axs[0].set_xlabel('Relative RA (arcsec)')
    axs[0].set_ylabel('Relative Dec (arcsec)')
    fig.colorbar(im1, ax=axs[0], label='Integrated Intensity')
    
    im2 = axs[1].imshow(map_y.T, origin='lower', extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axs[1].set_title('y-scan Map (Dec Scanning)')
    axs[1].set_xlabel('Relative RA (arcsec)')
    axs[1].set_ylabel('Relative Dec (arcsec)')
    fig.colorbar(im2, ax=axs[1], label='Integrated Intensity')
    
    im3 = axs[2].imshow(basket_map.T, origin='lower', extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axs[2].set_title('Basket Weave Map (Combined)')
    axs[2].set_xlabel('Relative RA (arcsec)')
    axs[2].set_ylabel('Relative Dec (arcsec)')
    fig.colorbar(im3, ax=axs[2], label='Integrated Intensity')
    
    plt.tight_layout()
    plt.savefig('nobeyama_fmlo_simulation.png', dpi=150)
    print("完了しました。画像は 'nobeyama_fmlo_simulation.png' として保存されました。")

if __name__ == '__main__':
    main()
