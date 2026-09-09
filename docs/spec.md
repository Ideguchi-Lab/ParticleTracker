# pt3d 実装リファレンス

この文書は、現在のコードが提供するAPIと処理内容を記載します。
以前の設計ドラフトにあった将来機能は、実装済み機能に含めていません。
導入・使用例は[User Guide](user_guide.md)、設定値は
[Configuration Reference](configuration.md)を参照してください。

## 構成と処理の流れ

| モジュール | 役割 |
|---|---|
| `pt3d.config` | Pydanticによる入力・検出・追跡・出力設定 |
| `pt3d.io` | ndarray、NPY、TIFF、Zarrの読み込みと軸順の正規化 |
| `pt3d.detect` | trackpyの`locate`、`batch`を使う3D粒子検出 |
| `pt3d.track` | 物理距離に対応する座標スケーリングと粒子のリンク |
| `pt3d.postprocess` | 速度による点の除去、短い軌跡の除去、統計量計算 |
| `pt3d.pipeline` | 検出から出力までの処理を統合 |
| `pt3d.export` | DataFrame、設定、実行情報の保存 |
| `pt3d.napari` | 表示用データ変換とQtウィジェット |
| `pt3d.synth` | 合成粒子画像・FBM軌跡の生成 |
| `pt3d.analysis` | MSD・異常拡散フィット・matplotlibによる描画 |

標準パイプラインは、入力取得 → 検出 → リンク → 速度フィルタ（任意）→
短い軌跡の除去 → 統計量計算 → 保存（任意）の順に実行します。
独立した`PreprocessConfig`や`pt3d.preprocess`はありません。
検出時の`preprocess=True`はtrackpyのbandpass前処理を有効にします。
背景補正やリサンプリングを別途行う場合は、入力前に実行してください。

## 入力と座標

内部の標準軸順は`(t, z, y, x)`です。位置列`z, y, x`の単位はピクセルです。
`VoxelSize(z_um, y_um, x_um)`は各軸のサンプリング間隔をµmで表し、
パイプライン設定では必須です。

| 入力方法 | 挙動 |
|---|---|
| `run_pipeline(array, config)` | 配列は`tzyx`または`zyx`が前提。3Dなら先頭に時間軸を追加 |
| `run_pipeline(path, config)` | `config.input.axis_order`を使って読み込み時に正規化 |
| `run_pipeline_streaming(iterator, config, n_frames=None)` | `zyx`の3D配列を順番に受け取り、0からフレーム番号を付与 |
| `load_array(data, axis_order="tzyx")` | 配列を`tzyx`に正規化。3D配列では`axis_order="zyx"`などを指定 |
| `load_data(source, axis_order="tzyx", dataset_key=None)` | 配列またはファイルから読み込み・正規化 |

`run_pipeline`への配列入力では`config.input.axis_order`を使いません。
異なる軸順の配列は`load_array`で変換してから渡してください。
3Dファイルの入力では`axis_order="zyx"`などの3軸指定が必要です。
`input.dtype`は設定として保存されるだけで、型変換は行われません。

Zarrのルートがgroupの場合は、`load_data`や`load_zarr`の`dataset_key`で
配列を指定してください。`InputConfig`には`dataset_key`がないため、
必要な配列を読み込んでからパイプラインへ渡します。
通常のファイル読み込みは配列全体をメモリに展開します。
ストリーミング版は画像を1フレームずつ処理しますが、検出点のDataFrameは
全フレーム分を保持してからリンクします。

## 検出API

`pt3d.detect`には以下の関数があります。

```python
detect_frame(volume, config, voxel_size=None)
detect_batch(frames, config, frame_range=None, voxel_size=None)
detect_streaming(frame_iterator, config, voxel_size=None)
detect_single_frame(frames, frame_index, config, voxel_size=None)
```

`DetectionConfig`では`diameter`と`diameter_um`のどちらか一方を指定します。
`diameter`は正の奇数3個の組`(dz, dy, dx)`です。
`diameter_um`を使う個別検出関数には`voxel_size`も渡してください。
パイプラインは設定内のvoxel sizeを自動で渡します。
µmからの変換は、各軸のvoxel sizeで割って整数に丸め、最小値を1とし、
偶数なら1を加えます。

`frame_range=(start, end)`は終端を含みません。
`detect_batch`の出力フレーム番号は元の入力配列に対応します。
ストリーミング版は、検出のないフレームも含めてiterator順に番号を付けます。
検出エラーが発生すると例外を送出し、失敗フレームをスキップして継続する
オプションはありません。

非空の検出結果は`z, y, x, mass`とtrackpy由来の特徴量を含みます。
`detect_frame`を除く関数は`frame`を追加します。
異方的なdiameterでは`size_z/y/x`や`ep_z/y/x`など、軸別の列が返ります。
空の結果には`size, ecc, signal, raw_mass, ep`を含む固定スキーマを使うため、
非空時と特徴量の列が一致するとは限りません。
`detect_single_frame`の空結果には`frame`列がありません。

## リンクと後処理

```python
from pt3d.track import link_detections, relabel_tracks
from pt3d.postprocess import postprocess, compute_track_stats

tracks = link_detections(detections, tracking_config, voxel_size)
tracks = postprocess(tracks, postprocess_config, voxel_size)
stats = compute_track_stats(tracks, voxel_size)
```

リンク時は各座標を`voxel_size / min(voxel_size)`でスケーリングし、
`search_range_um`も最小voxel sizeで割ってtrackpyに渡します。
返り値は元のピクセル座標を保持し、`particle`列が追加されます。

`memory`は検出欠測を許すフレーム数です。
`adaptive_stop`と`adaptive_step`は両方を指定するか、両方省略します。
`adaptive_stop`は粒子数ではなく、過大なサブネットで探索距離を縮小する際の
停止距離です。現在はµmから変換せず、スケーリング後の座標単位でそのまま
trackpyへ渡します。µmで指定したい閾値は最小voxel sizeで割ってください。

`min_track_length`は観測点数で判定し、欠測フレームは数えません。
既定値2では1点だけの軌跡が除かれます。
速度は3D変位をフレーム差で割ったµm/frameであり、µm/sではありません。
速度フィルタは閾値を超えた点を削除し、その後に短い軌跡を削除します。

`compute_track_stats`の列は次のとおりです。

| 列 | 意味 |
|---|---|
| `particle` | 軌跡ID |
| `length` | 観測点数 |
| `duration` | 最終frame − 最初のframe |
| `mean_velocity_um` | 観測点間の速度の算術平均、µm/frame |
| `max_velocity_um` | 最大速度、µm/frame |
| `total_displacement_um` | 始点と終点の直線距離、µm |

## 結果と出力

`run_pipeline(data, config)`とストリーミング版は`PipelineResult`を返します。
主な属性は`detections`、`tracks`、`track_stats`、`config`、`input_info`、
`start_time`、`end_time`です。`n_detections`、`n_tracks`、`duration_seconds`と
`summary()`も提供します。実行時間は自動エクスポートの前までを測定します。

`config.export`が指定されると、以下の4ファイルを保存します。

- `detections.parquet`または`detections.csv`
- `tracks.parquet`または`tracks.csv`
- `config.yaml`
- `run.json`

`track_stats`は自動保存されません。必要ならDataFrameから明示的に保存します。
個別の保存APIは`pt3d.pipeline.export_results(result, pipeline_config)`です。
`overwrite=False`では既存ファイルへの上書きを拒否します。
出力は順次行われ、途中で失敗した場合に先に保存したファイルは残ります。

`run.json`には設定、開始・終了時刻、実行時間、結果サマリー、Python・OSと
pt3d/trackpy/NumPy/pandasのバージョンを保存します。
通常入力の情報はsource・shape・dtype、ストリーミング入力ではsourceと
呼び出し側が渡したn_framesです。入力ハッシュ・mtime・全依存の一覧は収集しません。
失敗時のrun.jsonはパイプラインから自動保存されません。
`export_run_info(error=...)`を直接呼べば、渡したエラー文字列を保存できます。

## 拡散解析とシミュレーション

`analyze_diffusion(pipeline_result, analysis_config=None)`は観測点をµmへ変換し、
欠測をNaNで表す連続フレーム配列を作ります。最短軌跡の選別は観測点数、
解析結果の`n_frames`は欠測を含むフレーム範囲の長さです。
`interpolate_gaps=True`で線形補間し、FalseではMSDの各lagで有効な点対のみを使います。

モデルは`MSD = 6 * D * t**alpha`です。時間は秒、Dの単位はµm²/s^alphaです。
`fit_diffusion_exponent`の返り値は`(alpha, D)`で、常に振幅を6で割ります。
`analyze_single_track`は`(D, alpha, msd)`を返します。
成分別の1D MSDから得た係数をそのまま1D拡散係数として解釈しないでください。

最大lagの割合は個々の粒子に適用します。フィット範囲はlag 0を含むMSD配列長に
割合を掛け、整数化と最低点数の補正を行います。
正確な計算式は[MSDConfig](configuration.md#msdconfig)を参照してください。
ensemble MSDは別途、最短フレーム範囲の半分まで計算し、各粒子のMSDを等重みで平均します。

`plot_msd_loglog`には`MSDPlotConfig(show_fit=True)`をconfig引数に渡します。この設定は
`6 * mean_d * t`という傾き1の参考線を表示します。
凡例は`α=1 fit`ですが、推定alphaを使う曲線やensembleのフィットではありません。

合成軌跡はFBMの共分散行列のCholesky分解で生成します。
シミュレーションのDも一般化係数で、単位はµm²/s^(2H)、alpha=2Hです。
画像とともに返すground truth座標はピクセル単位の`[z, y, x]`です。
境界条件は座標の反射・周期的な折り返し・各時刻のクリップを提供します。
`absorbing`はクリップの名称であり、一度境界に到達した粒子を以降固定する処理では
ありません。境界処理後のMSDは無境界のFBMモデルから変わる場合があります。

## napari

Points用配列は`[t, z, y, x]`または`[z, y, x]`、Tracks用は
`[particle, t, z, y, x]`です。`to_napari_tracks`はparticle・frame順に整列します。
`get_napari_scale`は`(1, z_um, y_um, x_um)`などの表示スケールを返します。
レイヤ作成時に明示的に渡してください。

プラグインは`MainWidget`と`Track3DVisualizationWidget`を提供します。
MainWidgetでは検出・追跡・統計表示・統計のCSV/Parquet保存が可能です。
検出と追跡はGUIスレッドで同期実行され、バックグラウンドworkerや設定別の結果キャッシュは
実装されていません。検出・軌跡・設定・run情報の一括保存はPython APIを使います。

3Dウィジェットは描画方式・コントラスト・色・軌跡長・カメラ・時間操作を提供します。
`set_data(image=None, tracks=None, detections=None, voxel_size=None)`は
追跡メタデータと時間スライダー範囲を設定します。レイヤは作成せず、
`detections`引数は未使用です。画像・点・軌跡のレイヤは先に作成してください。

以下は現在の制約です。

- `NapariConfig`の名前とframe_rangeは保存されるだけで、自動適用されません。
- `show_current_position`と`show_current_frame_only`は描画に使われません。
- 個別Track IDのhighlight操作はログを出すだけです。
- `color_by="length"`、`"velocity"`、`"displacement"`にはtrack_statsが必要です。
  利用できないプロパティを選ぶとtrack IDによる着色へ戻ります。
- velocityによる着色は軌跡ごとの平均速度を使います。

## 設定ファイル・例外

`examples/run_from_yaml.py`はYAMLを`PipelineConfig.model_validate`で読み込みます。
リポジトリルートから実行し、スクリプト内のCONFIG_PATHなどを編集してください。
インストールされる専用CLIコマンドはありません。

設定モデルの不正値はPydanticの`ValidationError`になります。
`ConfigError`クラスは定義されていますが、現在の設定モデルは使用しません。
読み込み層では`DataError`、検出・リンクなどでは`ProcessingError`を使用します。
ファイルI/Oなどの例外がすべてこれらに包まれるわけではありません。
