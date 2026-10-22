# 段階1:1ビーム・1点観測の FMLO シミュレーション

野辺山45m望遠鏡での OTF + FMLO 観測を想定した大気雑音除去の検討のうち、段階1(1ビームで空の1点を見続ける場合)のシミュレーションです。
段階2(1ビームでスキャン)・段階3(マルチビーム)で再利用できるよう、処理を `fmlo_sim/` のモジュールに分けています。

「要確認」と書いた値は根拠を確認できていない仮の値です。結果を解釈する前に確認してください。

## 実行方法

```bash
pip install -r ../requirements.txt
cd step1_single_point

python run_step1.py                 # 既定設定で実行。outputs/step1/ に図と評価値を出力
python run_step1.py --method fa_ica # FA+ICA と除去なしのみを実行
python run_step1.py --method pca    # PCA と除去なしのみを実行
python run_step1.py --fa-factors 3 --fa-remove 0,1 # FA+ICA の因子数と除去成分を変更
```

設定は `config/default.yaml` を編集するか、CLI オプション（`--fa-factors`, `--fa-remove`, `--fa-mean-mode`）または `--set` で一時的に上書きします。

```bash
python run_step1.py --outdir outputs/offset50 --set noise.atmosphere.offset_k=50
python run_step1.py --set removal.fa_ica.n_factors=3 --set removal.fa_ica.remove_indices="[0, 1]"
```

同じ設定ファイル(同じ `seed`)なら結果は再現します。乱数は成分ごと(変調・熱雑音・大気・ICA)に独立なので、ある成分の設定を変えても他の成分の乱数列は変わりません。

## 処理の流れ

| 手順 | 内容 | モジュール |
|---|---|---|
| 1 | 正解キューブ(RA × Dec × 周波数、単位 K)を作る。ガウシアンのボール1つ | `cube.py` |
| 2 | 指定ピクセル(既定は天体中心)のスペクトルを取り出す | `cube.py` |
| 3 | FMLO 変調:各時刻で観測窓をずらし、広い周波数グリッドから窓を切り出す | `modulation.py` |
| 4 | 雑音を成分ごとに作って足す(熱雑音、大気) | `noise.py` |
| 5 | 相関雑音除去: **PCA** または **FA+ICA** により相関成分を除去する | `removal.py` |
| 6 | 記録したシフト量で復調し、時間平均して復元スペクトルを得る | `modulation.py` |
| 7 | 正解と比べて評価値を出す | `evaluation.py` |
| 8 | 各段階の図を出す | `plotting.py` |

手順 1〜4 は `pipeline.simulate()`、5〜7 は `pipeline.reduce()` / `pipeline.reduce_fa_ica()`、一括実行は `pipeline.reduce_all_modes()` にまとめてあります。設定の読み込みは `config.py`、単位変換は `units.py` です。

### モデルの定義と仮定

- **周波数グリッド**:キューブの周波数軸は分光計の帯域より広く、両側に「変調の半幅 + 余白」チャンネルずつ余分に取ります。チャンネル幅は分光計と同じなので、窓の切り出しは整数チャンネルのスライスです。`np.roll` は使っておらず、窓がグリッドからはみ出す設定はエラーになります。
- **シフト量の符号**:正のシフトは窓が高周波側へ動くことを表します。
- **速度と周波数**:Δν = ν Δv / c(radio 定義)。輝線幅は FWHM、天体の空間サイズは σ で指定します(変数名で区別、変換は `units.fwhm_to_sigma` / `sigma_to_fwhm`)。
- **ビーム**:望遠鏡ビームの畳み込みは入れていません。キューブのピクセル値をそのまま観測値とします。
- **熱雑音**:ΔT = Tsys / sqrt(Δν_ch × Δt) の標準偏差で、各時刻・各チャンネル独立に生成します。OFF 点の差し引きがない前提で、√2 の係数は付けていません。
- **大気**:`(offset_k + a(t)) × s(ν)`。a(t) は全チャンネル共通の時間変動で、パワースペクトルが f^(-α) に従う有色雑音です(平均 0、標準偏差 `rms_k`)。s(ν) は **空の周波数** の関数 `1 + slope_frac × (ν − ν_c) / (B/2)` で、天体信号と同じ窓で切り出します。したがって窓をずらすと、大気の傾きも一緒にチャンネル上を動きます。
- **復調後の評価範囲**:全時刻で観測された周波数範囲(ヒット数 = 時間サンプル数)だけで評価します。
- **輝線のない領域**:輝線中心から ±(`line_free_n_fwhm` × FWHM) の外側です。積分強度はその内側で積分します。

## 雑音除去手法

同じ模擬観測データに対して、公平に比較できるよう複数の雑音除去手法を適用します。

### 1. PCA による除去

sklearn の PCA は必ず各チャンネルの時間平均を引くため、ここでは numpy の SVD で実装し、`mean_mode` で平均の扱いを明示的に選びます。

| `mean_mode` | 処理 |
|---|---|
| `none`(**PCA既定**) | 平均を引かずに SVD し、上位成分を除去する(原点まわりの PCA) |
| `channel_mean` | 各チャンネルの時間平均を引いてから SVD し、上位成分を除去する。引いた平均は戻さない |

### 2. FA+ICA による除去

因子分析 (Factor Analysis: FA) により共通因子空間を推定し、独立成分分析 (Independent Component Analysis: ICA) により非ガウス性に基づいて成分を回転分離します。

1. **FA**: 中心化データ $X_c = X - \boldsymbol{\mu}$ に対し、$K$ 個の共通因子を推定:
   $$X_c \approx Z L^T$$
   ($Z \in \mathbb{R}^{T \times K}$: 因子得点, $L \in \mathbb{R}^{C \times K}$: 因子負荷行列)
2. **FastICA**: 因子得点 $Z$ を独立成分 $S \in \mathbb{R}^{T \times K}$ と混合行列 $A \in \mathbb{R}^{K \times K}$ に分解 ($Z \approx S A^T$)。
3. **元空間への再構成**:
   有効スペクトルプロファイル $M = A^T L \in \mathbb{R}^{K \times C}$ を介して、選択した除去対象成分 $J \subseteq \{0, \dots, K-1\}$ のみを元データ空間 $(T, C)$ へ再構成して差し引きます:
   $$X_{\text{removed, fluc}} = S_{:, J} M_{J, :}$$
   - `mean_mode="channel_mean"`: $X_{\text{removed}} = X_{\text{removed, fluc}} + \mathbf{1} \boldsymbol{\mu}^T$, $X_{\text{cleaned}} = X - X_{\text{removed}}$
   - `mean_mode="none"`: $X_{\text{removed}} = X_{\text{removed, fluc}}$, $X_{\text{cleaned}} = X - X_{\text{removed}}$ (平均は保持)

> **重要 (全成分除去時の性質)**:
> 推定した全因子 $J = \{0, \dots, K-1\}$ をすべて除去する場合、$S M = (S A^T) L = Z L^T$ となり、ICA の回転は相殺されて **FA 単体で全共通因子を除去した結果と数学的に完全に一致** します。
> FA+ICA の真の意義は、「因子空間の中から大気雑音成分を選択的に分離し、天体信号を巻き込まずに除去できるか」という成分選択にあります。

## 比較実験結果

### 既定設定 (seed = 20261005, offset_k = 0 K)

熱雑音だけを時間平均した場合の期待 RMS は 0.0370 K です。

| 手法・処理 | 因子数 / 成分数 | 除去成分 | 輝線なし残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|---|---|
| `no_pca` (除去なし) | 0 | — | 0.0372 | 1.0005 | 0.9838 |
| PCA (`none`) | 1 | 主成分1 | 0.0394 | 0.9876 | 0.9336 |
| PCA (`channel_mean`) | 1 | 主成分1 | 0.0429 | 0.8309 | 0.4327 |
| **FA+ICA (`channel_mean`)** | 2 | Comp 0 | **0.0430** | **0.8308** | **0.4337** |
| FA+ICA (`channel_mean`) | 2 | Comp 0, 1 (全除去) | 0.0389 | 0.7324 | 0.2189 |

- `offset_k = 0 K` の場合、大気の時間変動の平均は 0 のため、`no_pca` でも時間平均だけで大気が消え、最も正解に近くなります。
- `channel_mean` (PCA / FA+ICA) はチャンネル時間平均を引くため、変調でならされた天体輝線の時間平均も一緒に差し引かれ、積分強度が約 57% 削られます。
- FA+ICA で Comp 0 のみを除去した場合、PCA `channel_mean` (k=1) と同等の性能を示します。一方、Comp 1 も除去すると残差 RMS は 0.0389 K に低下しますが、天体信号がさらに過剰除去され積分強度比が 0.2189 まで激減します。

### 大気オフセット 50 K (seed = 20261005)

| 手法・処理 | 因子数 / 成分数 | 除去成分 | 輝線なし残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|---|---|
| `no_pca` (除去なし) | 0 | — | 50.0753 | 51.0125 | 188.0084 |
| PCA (`none`) | 1 | 主成分1 | 0.0428 | 0.8318 | 0.4381 |
| PCA (`channel_mean`) | 1 | 主成分1 | 0.0423 | 0.8236 | 0.4115 |
| **FA+ICA (`channel_mean`)** | 2 | Comp 0 | **0.0429** | **0.8295** | **0.4315** |

- 大気オフセットが存在する場合、`no_pca` ではオフセットがそのまま残り観測不能になります。
- PCA `none` では第1主成分がオフセットを吸収しますが、輝線の時間平均も一部巻き込みます。
- FA+ICA は平均を `channel_mean` として処理し、変動成分のうち大気成分 (Comp 0) を除去することで、PCA と同等水準の復元を達成します。

### 複数 seed (5 seeds: 20261005–20261009) での集計結果

同一 seed の観測データに対して各手法を一括適用した集計値 (mean ± std):

#### Offset: 0 K
| 手法・処理 | 残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|
| `no_pca` | 0.0370 ± 0.0004 | 0.9931 ± 0.0164 | 0.9923 ± 0.0053 |
| PCA (`none`) | 0.0392 ± 0.0004 | 0.9787 ± 0.0173 | 0.9392 ± 0.0066 |
| PCA (`channel_mean`) | 0.0429 ± 0.0005 | 0.8248 ± 0.0231 | 0.4369 ± 0.0063 |
| **FA+ICA (`channel_mean`, rem=[0])** | 0.0429 ± 0.0006 | 0.8248 ± 0.0231 | 0.4378 ± 0.0053 |

#### Offset: 50 K
| 手法・処理 | 残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|
| `no_pca` | 50.0761 ± 0.0007 | 51.0050 ± 0.0164 | 188.0168 ± 0.0053 |
| PCA (`none`) | 0.0428 ± 0.0005 | 0.8263 ± 0.0229 | 0.4423 ± 0.0051 |
| PCA (`channel_mean`) | 0.0424 ± 0.0005 | 0.8196 ± 0.0243 | 0.4258 ± 0.0117 |
| **FA+ICA (`channel_mean`, rem=[0])** | 0.0428 ± 0.0005 | 0.8237 ± 0.0238 | 0.4356 ± 0.0079 |

## パラメータ一覧

すべて `config/default.yaml` で定義します。

| パラメータ | 意味 | 単位 | 既定値 | 根拠 |
|---|---|---|---|---|
| `seed` | 乱数シード | — | 20261005 | 任意 |
| `source.sigma_arcsec` | 天体の空間サイズ(ガウシアンの σ) | arcsec | 50 | 打ち合わせの指定。FWHM では 117.7 arcsec |
| `source.line_fwhm_kms` | 輝線幅(速度の FWHM) | km/s | 50 | 打ち合わせの指定。150 GHz で 25.0 MHz |
| `source.center_freq_ghz` | 輝線の中心周波数 | GHz | 150 | 打ち合わせの指定。125–163 GHz の範囲外はエラー |
| `source.peak_k` | ピーク強度 | K | 1.0 | **要確認**(仮の値) |
| `source.center_ra_arcsec`, `center_dec_arcsec` | 天体中心の位置 | arcsec | 0, 0 | マップ中心 |
| `cube.half_extent_arcsec` | マップの半幅 | arcsec | 200 | 計算上の都合(±4σ) |
| `cube.pixel_arcsec` | ピクセルサイズ | arcsec | 10 | 計算上の都合。**段階2以降はビームサイズに合わせて要確認** |
| `cube.margin_ch` | 変調幅の外側に取る周波数方向の余白 | ch | 50 | 計算上の都合 |
| `pointing.ra_arcsec`, `dec_arcsec` | スペクトルを取り出す位置 | arcsec | null(天体中心) | — |
| `spectrometer.n_channels` | 分光計のチャンネル数 | ch | 4096 | **要確認**(SAM45 を想定。資料で未確認) |
| `spectrometer.channel_width_mhz` | チャンネル幅 | MHz | 0.48828125(= 2000/4096) | **要確認**(SAM45 を想定。資料で未確認) |
| `spectrometer.center_freq_ghz` | シフト 0 のときの帯域中心 | GHz | 150 | 輝線中心に合わせた |
| `modulation.pattern` | 変調パターン(`zigzag` / `random`) | — | zigzag | — |
| `modulation.dt_s` | 変調(=サンプリング)の時間間隔 | s | 0.1 | 打ち合わせの指定 |
| `modulation.half_width_ch` | 変調の半幅。シフト量は ± この値 | ch | 150(± 約 73 MHz) | **要確認**(仮の値) |
| `modulation.zigzag_step_ch` | ジグザグの1サンプルあたりのシフト量 | ch | 10 | **要確認**(仮の値)。周期は 4 × 半幅 / この値 = 60 サンプル = 6 秒 |
| `observation.duration_s` | 観測時間 | s | 60(600 サンプル) | **要確認**(仮の値) |
| `noise.thermal.enabled` | 熱雑音を入れるか | — | true | — |
| `noise.thermal.tsys_k` | システム雑音温度 | K | 200 | 打ち合わせの指定 |
| `noise.atmosphere.enabled` | 大気成分を入れるか | — | true | — |
| `noise.atmosphere.model` | 大気雑音モデル(`laminar`: 層流 / `turbulent` or `colored`: 乱流) | — | laminar | 穏やかな低周波変動 |
| `noise.atmosphere.alpha` | 時間方向パワースペクトル f^(-α) の指数(colored の場合) | — | 1.0 | 打ち合わせの例(1、2)。**実際の値は要確認** |
| `noise.atmosphere.rms_k` | 時間変動の標準偏差 | K | 1.0 | **要確認**(仮の値) |
| `noise.atmosphere.offset_k` | 時間変動しない定数オフセット | K | 0 | **実際の大気の値は要確認** |
| `noise.atmosphere.slope_frac` | 周波数方向の傾き(シフト 0 の帯域の両端で ± この値) | — | 0.1 | **要確認**(仮の値) |
| `removal.n_components` | PCA で除去する成分数(0 で除去なし) | — | 1 | 仮の値 |
| `removal.mean_mode` | PCA の平均の扱い | — | none | 上の節を参照 |
| `removal.fa_ica.n_factors` | FA の共通因子数 | — | 2 | 初期比較値 |
| `removal.fa_ica.mean_mode` | FA+ICA の平均の扱い | — | channel_mean | channel_mean または none |
| `removal.fa_ica.remove_indices` | 除去対象の ICA 成分インデックス | — | [0] | 0-based リスト |
| `removal.fa_ica.random_state` | ICA 初期値シード | — | 42 | — |
| `removal.fa_ica.max_iter` | ICA 最大反復回数 | — | 500 | — |
| `removal.fa_ica.tol` | ICA 収束許容値 | — | 0.0001 | — |
| `evaluation.line_free_n_fwhm` | 輝線なし領域の境界(FWHM の何倍か) | — | 2.0 | 解析上の選択 |

既定値での派生量:帯域 2000 MHz、チャンネルあたりの速度幅 0.976 km/s、1サンプルの熱雑音 0.905 K、キューブの周波数チャンネル数 4496。

## 出力

`outputs/` は git 管理外です。`run_step1.py` は `outputs/step1/` に次を出力します。

| ファイル | 内容 |
|---|---|
| `cube.npz` | 正解キューブ(`cube[i_ra, i_dec, i_freq]` [K]、`ra_arcsec`、`dec_arcsec`、`freq_ghz`、`meta`)。`cube.load_cube()` で読める |
| `timestream.npz` | シフト量、天体信号のみ・各雑音成分・観測データの時間×チャンネル配列、各処理の復元スペクトル |
| `metrics.json` | 設定と、各手法の評価値および診断量 |
| `metrics_comparison.csv` | 評価値の比較表 |
| `fig01_cube.png` | 積分強度マップと取り出したスペクトル |
| `fig02_shift_pattern.png` | 変調パターン(時間 vs シフト量) |
| `fig03_modulated_signal.png` | 変調後の天体信号のみ(時間×チャンネル) |
| `fig04_noise_components.png` | 雑音の各成分と観測データ全体 |
| `fig05_pca_removal.png` | PCA で除去した成分、除去後のデータ、主成分のスペクトル |
| `fig05_fa_ica_components.png` | FA+ICA の成分診断図 (除去成分、除去後データ、ICA時間波形、有効スペクトルプロファイル) |
| `fig06_restored_spectrum.png` | 復元スペクトルと正解スペクトル、残差 (除去なし、PCA、FA+ICA を重ね描き) |
| `fig07_power_spectra.png` | 除去前後の時間方向パワースペクトル(輝線が一度も入らないチャンネル) |
