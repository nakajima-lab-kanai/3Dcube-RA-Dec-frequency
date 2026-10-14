# 段階1:1ビーム・1点観測の FMLO シミュレーション

野辺山45m望遠鏡での OTF + FMLO 観測を想定した大気雑音除去の検討のうち、段階1(1ビームで空の1点を見続ける場合)のシミュレーションです。
段階2(1ビームでスキャン)・段階3(マルチビーム)で再利用できるよう、処理を `fmlo_sim/` のモジュールに分けています。

「要確認」と書いた値は根拠を確認できていない仮の値です。結果を解釈する前に確認してください。

## 実行方法

```bash
pip install -r ../requirements.txt
cd step1_single_point

python run_step1.py                 # 既定設定で実行。outputs/step1/ に図と評価値を出力
python run_step1.py --plot3d        # 加えてキューブの3D表示 fig08_cube3d.html を出力(plotly が必要)
```

3D 表示は plotly の等値面で、ブラウザで回転・拡大できる HTML(plotly.js を埋め込んだ自己完結の1ファイル、約 10 MB)です。plotly は**任意依存**で、`pip install plotly` で別途入れます(`--plot3d` を付けない通常の実行には不要)。保存済みの `cube.npz` からも単独で描けます。

```bash
python -m fmlo_sim.plotting3d outputs/step1/cube.npz --out cube3d.html
```

設定は `config/default.yaml` を編集するか、`--half-width-ch` や `--zigzag-step-ch`、`--atmosphere-model`、または `--set` で一時的に上書きします。

```bash
python run_step1.py --half-width-ch 150 --zigzag-step-ch 10 --atmosphere-model laminar
python run_step1.py --outdir outputs/offset50 --set noise.atmosphere.offset_k=50 --set removal.n_components=2
python run_step1.py --set modulation.pattern=random
python run_step1.py --set source.line_fwhm_kms=100 --half-width-ch 300
```

同じ設定ファイル(同じ `seed`)なら結果は再現します。乱数は成分ごと(変調・熱雑音・大気)に独立なので、ある成分の設定を変えても他の成分の乱数列は変わりません。

## 処理の流れ

| 手順 | 内容 | モジュール |
|---|---|---|
| 1 | 正解キューブ(RA × Dec × 周波数、単位 K)を作る。ガウシアンのボール1つ | `cube.py` |
| 2 | 指定ピクセル(既定は天体中心)のスペクトルを取り出す | `cube.py` |
| 3 | FMLO 変調:各時刻で観測窓をずらし、広い周波数グリッドから窓を切り出す | `modulation.py` |
| 4 | 雑音を成分ごとに作って足す(熱雑音、大気) | `noise.py` |
| 5 | PCA で上位成分を除去する | `removal.py` |
| 6 | 記録したシフト量で復調し、時間平均して復元スペクトルを得る | `modulation.py` |
| 7 | 正解と比べて評価値を出す | `evaluation.py` |
| 8 | 各段階の図を出す | `plotting.py` |

手順 1〜4 は `pipeline.simulate()`、5〜7 は `pipeline.reduce()` にまとめてあります。設定の読み込みは `config.py`、単位変換は `units.py` です。

### モデルの定義と仮定

- **周波数グリッド**:キューブの周波数軸は分光計の帯域より広く、両側に「変調の半幅 + 余白」チャンネルずつ余分に取ります。チャンネル幅は分光計と同じなので、窓の切り出しは整数チャンネルのスライスです。`np.roll` は使っておらず、窓がグリッドからはみ出す設定はエラーになります。
- **シフト量の符号**:正のシフトは窓が高周波側へ動くことを表します。
- **速度と周波数**:Δν = ν Δv / c(radio 定義)。輝線幅は FWHM、天体の空間サイズは σ で指定します(変数名で区別、変換は `units.fwhm_to_sigma` / `sigma_to_fwhm`)。
- **ビーム**:望遠鏡ビームの畳み込みは入れていません。キューブのピクセル値をそのまま観測値とします。
- **熱雑音**:ΔT = Tsys / sqrt(Δν_ch × Δt) の標準偏差で、各時刻・各チャンネル独立に生成します。OFF 点の差し引きがない前提で、√2 の係数は付けていません。
- **大気**:`(offset_k + a(t)) × s(ν)`。a(t) は全チャンネル共通の時間変動で、パワースペクトルが f^(-α) に従う有色雑音です(平均 0、標準偏差 `rms_k`)。s(ν) は **空の周波数** の関数 `1 + slope_frac × (ν − ν_c) / (B/2)` で、天体信号と同じ窓で切り出します。したがって窓をずらすと、大気の傾きも一緒にチャンネル上を動きます。
- **復調後の評価範囲**:全時刻で観測された周波数範囲(ヒット数 = 時間サンプル数)だけで評価します。
- **輝線のない領域**:輝線中心から ±(`line_free_n_fwhm` × FWHM) の外側です。積分強度はその内側で積分します。

### 雑音成分の差し替え・追加

`NoiseComponents` に `(n_time, n_channels)` の配列を名前付きで `add()` します。

- 大気の時間変動を外部の時系列(将来は Gato)に替える場合は、`noise.atmosphere_noise()` の `timeseries_k` にその時系列を渡します。
- チャンネル(IF)側に固定された成分(装置由来の雑音)は今回実装していません。追加する場合は、窓で切り出さずに `(n_time, n_channels)` の配列をそのまま `add("instrument", tod)` します。

## PCA の平均の扱い

`sklearn.decomposition.PCA` は必ず各チャンネルの時間平均を引くため、ここでは numpy の SVD で実装し、`removal.mean_mode` で明示的に選びます。

| `mean_mode` | 処理 |
|---|---|
| `none`(**既定として採用**) | 平均を引かずに SVD し、上位成分を除去する |
| `channel_mean` | 各チャンネルの時間平均を引いてから SVD し、上位成分を除去する。引いた平均は戻さない(sklearn の `X - inverse_transform(transform(X))` と同じ。テストで一致を確認) |

`run_step1.py` は設定の `mean_mode` にかかわらず、同じ観測データに対して `none`、`channel_mean`、`no_pca`(PCA も平均除去もせず復調だけ)の3通りを評価し、`metrics.json` と `metrics_comparison.csv` に並べて出力します。図 5 は設定で選んだモード、図 6・7 は両モードを重ねて描きます。

### 既定設定での結果(seed = 20261005、除去成分数 1)

熱雑音だけを時間平均した場合の期待 RMS は 0.037 K です。

| 処理 | 輝線なし領域の残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|
| `none` | 0.039 | 0.98 | 0.93 |
| `channel_mean` | 0.043 | 0.83 | 0.43 |
| `no_pca` | 0.037 | 1.00 | 0.98 |

- `channel_mean` は輝線を大きく削ります。変調でならされた輝線の時間平均(幅が変調幅程度の台地)が平均スペクトルとして引かれ、復調後に輝線の両側が負にへこむためです。
- `none` でも積分強度が約 7 % 減ります。第1成分(ほぼ平らなスペクトル)が、各時刻の輝線の帯域平均を大気の変動と区別できずに吸収するためです。
- **この設定では `no_pca` が最も正解に近くなります。** 大気の時間変動 a(t) は平均がちょうど 0 になるよう作っているので、1点を見続けて時間平均するだけで消えるためです。PCA の効果は時間平均後のスペクトルではなく、時間方向のパワースペクトル(図 7)に現れます。定数オフセットがある場合は次節のとおり状況が変わります。
- ピーク比は1チャンネルの値の比なので、熱雑音による ±4 % 程度のばらつきを含みます。

### 大気の定数オフセットと平均モードの組み合わせ

`noise.atmosphere.offset_k` を 0 以外にしても最後まで動きます(テストあり)。既定設定から `offset_k` と除去成分数だけを変えた結果です。

| `offset_k` [K] | 成分数 | 処理 | 残差 RMS [K] | ピーク比 | 積分強度比 |
|---|---|---|---|---|---|
| 0 | 1 | `none` | 0.039 | 0.98 | 0.93 |
| 0 | 1 | `channel_mean` | 0.043 | 0.83 | 0.43 |
| 0 | 2 | `none` | 0.041 | 0.83 | 0.60 |
| 0 | 2 | `channel_mean` | 0.037 | 0.68 | 0.10 |
| 50 | 1 | `none` | 0.043 | 0.83 | 0.44 |
| 50 | 1 | `channel_mean` | 0.043 | 0.83 | 0.43 |
| 50 | 2 | `none` | 0.037 | 0.68 | 0.09 |
| 50 | 2 | `channel_mean` | 0.037 | 0.68 | 0.09 |
| 50 | — | `no_pca` | 50.1 | 51.0 | 188 |

- オフセットが大きいと、`none` の第1成分はほぼ平均スペクトルそのものになり、`none` と `channel_mean` の結果はほとんど同じになります。`none` を選んでも輝線の削れは避けられません。
- オフセットがあると `no_pca` ではオフセットがそのまま残るので、何らかの除去が必要です。
- 成分数を増やすと輝線なし領域の残差は熱雑音の水準まで下がりますが、輝線の損失も大きくなります。残差 RMS だけで成分数を選ぶと輝線を失います。
- 大気の形を空の周波数に固定したため、窓をずらすと `offset_k × slope_frac × (シフト量 / 帯域の半幅)` の共通成分がシフト量に同期して現れます。オフセット 50 K・既定の傾きと変調幅では最大 ±0.37 K で、大気成分は厳密には2成分(ランク2)になります。

以上は単純な PCA 除去の性質を示したもので、輝線の損失を抑える方法(輝線モデルを先に引いてから PCA を繰り返すなど)は段階1では実装していません。

輝線幅を変える場合は `--set source.line_fwhm_kms=100` のように指定します。輝線が太いほど輝線の損失は増えます。

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
| `cube.margin_ch` | 変調幅の外側の余白 | ch | 50 | 計算上の都合 |
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
| `evaluation.line_free_n_fwhm` | 輝線なし領域の境界(FWHM の何倍か) | — | 2.0 | 解析上の選択 |
| `visualization.cube3d_half_range_n_fwhm` | 3D 表示で、周波数方向を輝線中心 ±(この値 × FWHM) に切り出す | — | 3.0 | 見た目の選択(変調の余白を除くため) |
| `visualization.cube3d_iso_fractions` | 等値面の値(切り出し範囲のピーク値に対する割合) | — | [0.2, 0.5, 0.8] | 見た目の選択 |
| `visualization.cube3d_opacity` | 等値面の透明度 | — | 0.35 | 見た目の選択 |
| `visualization.cube3d_max_freq_points` | 3D 表示の周波数方向の点数の上限(超えると間引く) | — | 40 | HTML のサイズを抑えるため |

既定値での派生量:帯域 2000 MHz、チャンネルあたりの速度幅 0.976 km/s、1サンプルの熱雑音 0.905 K、キューブの周波数チャンネル数 4496。

## 出力

`outputs/` は git 管理外です。`run_step1.py` は `outputs/step1/` に次を出力します。

| ファイル | 内容 |
|---|---|
| `cube.npz` | 正解キューブ(`cube[i_ra, i_dec, i_freq]` [K]、`ra_arcsec`、`dec_arcsec`、`freq_ghz`、`meta`)。`cube.load_cube()` で読める |
| `timestream.npz` | シフト量、天体信号のみ・各雑音成分・観測データの時間×チャンネル配列、各処理の復元スペクトル |
| `metrics.json` | 設定と、3通りの処理の評価値 |
| `metrics_comparison.csv` | 評価値の比較表 |
| `fig01_cube.png` | 積分強度マップと取り出したスペクトル |
| `fig02_shift_pattern.png` | 変調パターン(時間 vs シフト量) |
| `fig03_modulated_signal.png` | 変調後の天体信号のみ(時間×チャンネル) |
| `fig04_noise_components.png` | 雑音の各成分と観測データ全体 |
| `fig05_pca_removal.png` | PCA で除去した成分、除去後のデータ、主成分のスペクトル |
| `fig06_restored_spectrum.png` | 復元スペクトルと正解スペクトル、残差 |
| `fig07_power_spectra.png` | PCA 前後の時間方向パワースペクトル(輝線が一度も入らないチャンネル) |
| `fig08_cube3d.html` | キューブの3D等値面(`--plot3d` のときだけ)。軸は RA・Dec オフセット [arcsec] と視線速度 [km/s]、hover で周波数 [GHz] と強度 [K]。破線は観測した点 |

評価値の意味は次のとおりです。

| キー | 意味 |
|---|---|
| `line_free_residual_rms_k` | 輝線のない領域での(復元 − 正解)の RMS [K] |
| `line_free_residual_mean_k` | 同じ領域での残差の平均 [K](ベースラインのずれ) |
| `peak_ratio` | 正解のピークのチャンネルでの 復元 / 正解 |
| `integrated_ratio` | 積分強度 [K km/s] の 復元 / 正解 |
| `expected_thermal_rms_k` | 熱雑音だけを時間平均したときの期待 RMS [K] |
