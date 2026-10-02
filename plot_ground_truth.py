import numpy as np
import matplotlib.pyplot as plt

def plot_ground_truth():
    # nobeyama_fmlo_simulation.py の設定に合わせたパラメータ
    extent = [-150, 150, -150, 150] # 空間範囲 (arcsec)
    spatial_size = 50.0             # 天体の空間サイズ (arcsec)
    
    # 規則的な2次元グリッドの生成
    ra_grid = np.linspace(extent[0], extent[1], 300)
    dec_grid = np.linspace(extent[2], extent[3], 300)
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    
    # 真の天体空間分布（ガウシアンボール）を計算
    # FWHM = 50 arcsec から sigma への変換
    sigma = spatial_size / 2.355
    spatial_profile = np.exp(-((RA)**2 + (DEC)**2) / (2 * sigma**2))
    
    # 描画
    plt.figure(figsize=(7, 6))
    # シミュレーションの出力マップと同じ 'magma' カラーマップを使用
    im = plt.imshow(spatial_profile, origin='lower', extent=extent, cmap='magma')
    
    plt.title('Ground Truth (Target Sky Model - 50 arcsec)')
    plt.xlabel('Relative RA (arcsec)')
    plt.ylabel('Relative Dec (arcsec)')
    plt.colorbar(im, label='Normalized Intensity')
    
    # 半値幅(FWHM)を示す円を追加（オプション: 視覚的な目安として）
    circle = plt.Circle((0, 0), spatial_size/2, color='white', fill=False, linestyle='--', alpha=0.7, label='FWHM (50 arcsec)')
    plt.gca().add_patch(circle)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('ground_truth_map.png', dpi=150)
    print("完了しました。真の天体分布マップを 'ground_truth_map.png' として保存しました。")

if __name__ == '__main__':
    plot_ground_truth()
