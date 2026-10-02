# 3Dcube-RA-Dec-frequency

nakajima'slab 3Dcube coding task.

本リポジトリ（ディレクトリ）には、FMLO（周波数変調局部発振器）法を用いた電波天文学のマッピング観測のシミュレーションと、そのデータ解析（DE:MIST法等）を行うためのPythonスクリプトが含まれています。

## ファイル構成

- `nobeyama_fmlo_simulation.py`: 野辺山45m望遠鏡の150GHz帯受信機を想定し、相関ノイズと熱雑音を個別にモデル化・分離処理（PCA）を行う最新版のシミュレーションスクリプト。
- `fmlo_simulation.py`: 基本的なFMLO法とBasket Weave処理のデモンストレーションを行う初期バージョンのスクリプト。
- `nobeyama_fmlo_simulation.png` / `fmlo_basket_weave_results.png`: それぞれのスクリプトが出力した2次元積分強度マップ（x-scan, y-scan, Basket Weave）の画像。

## コードの詳細解説（関数単位）

ここでは主となる `nobeyama_fmlo_simulation.py` の各関数の役割について説明します。

### 1. `generate_scan_path(scan_type, n_lines, points_per_line, extent)`
観測時のスキャン軌跡（座標情報）を生成する関数です。
- **機能**: `scan_type` で指定された方向（'x'の場合はRA方向、'y'の場合はDec方向）に沿って、ラスタースキャン（ジグザグのスキャン軌跡）をシミュレートし、時間経過に伴う座標（`ra`, `dec`）の1次元配列（タイムストリーム）を返します。

### 2. `generate_sky_model(ra, dec, n_channels, center_ch, spatial_size, velocity_width_ch)`
観測対象となる天体の真のシグナル（模擬データ）を生成する関数です。
- **機能**: 3次元データキューブは使わず、指定された `ra`, `dec` のタイムストリーム上で計算を行います。空間的にはガウシアンボール（2次元ガウス分布、例：50 arcsec幅）、スペクトル（周波数）方向にもガウス分布（例：50 km/s幅に相当するチャンネル幅）を持つ輝線天体をモデル化します。

### 3. `apply_fmlo_modulation(signal, shift_pattern)`
FMLO法の中核である「周波数変調」をシミュレートする関数です。
- **機能**: 観測中（タイムステップごと）にLO周波数を高速に変化させる操作を、周波数チャンネルの循環シフト（`np.roll`）として実装しています。これにより、天体の輝線成分が時間に対して非相関な動きを持つようになります。

### 4. `generate_noise(n_time, n_channels)`
観測に乗ってくる各種ノイズを生成する関数です。
- **機能**: 大気放射や装置のベースライン変動などに起因する「時間・チャンネル方向に強い相関を持つ低周波ノイズ（相関ノイズ）」と、ランダムな「熱雑音（ホワイトノイズ）」を合成して返します。

### 5. `fmlo_demodulate(data, shift_pattern)`
データ解析の段階で行う「復調（逆シフト）」処理を担う関数です。
- **機能**: `apply_fmlo_modulation` で与えたシフト量とは逆のシフト（`-shift`）を各タイムステップで適用します。これにより、FMLOで散らされた天体の輝線信号を本来のチャンネル位置に揃え（積分可能な状態にし）ます。

### 6. `grid_data(ra, dec, values, extent, grid_size)`
タイムストリームデータを2次元画像（マップ）に再構成する関数です。
- **機能**: 不規則に並ぶスキャン軌跡上のデータ（`ra`, `dec`, `values`）を、`scipy.interpolate.griddata` を用いて等間隔な2次元グリッド上に内挿（グリッディング）し、積分強度マップを作成します。

### 7. `main()`
シミュレーションから解析までのパイプライン全体を実行するメイン関数です。以下の手順を制御します。
1. **スキャン軌跡の生成**（x-scan, y-scan）
2. **天体シグナルの生成**
3. **FMLO変調の適用**
4. **ノイズの付加**（観測模擬データの完成）
5. **PCAを用いた相関ノイズ除去（DE:MIST法）**: `sklearn.decomposition.PCA` を用い、分散の大きい大気などの共通成分（ベースライン）を推定して除去します。
6. **信号の復調**
7. **輝線成分の積分とマップ化**
8. **Basket Weave処理**: 直交する x-scan と y-scan のマップを統合（本コードでは単純平均）し、スキャンに起因するストライプノイズを低減させます。
9. **Matplotlibによる画像の出力・保存**
