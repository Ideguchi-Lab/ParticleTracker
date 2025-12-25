以下に、あなたが作成予定の「三次元時系列（3D+t）粒子検出・追跡ライブラリ（trackpy + napari パイプライン）」の**仕様書ドラフト**を、実装に直結する粒度でまとめます。
（Markdown形式。リポジトリの `SPEC.md` / `docs/spec.md` としてそのまま配置できる想定です。）

注意事項
pythonのパッケージマネージャーとしてはuv を使用してください。
コード中のコメントやコミットメッセージなどは英語で書いてください。

napariは `napari[all]` でインストールしてください。
trackpyは`trackpy`でインストールしてください。

---

# 3D+t Particle Tracking Library 仕様書（trackpy + napari）

* 文書種別：仕様書（ドラフト）
* バージョン：v0.1
* 作成日：2025-12-25
* 対象：顕微鏡 3D+t データに対する particle 検出（detection）とフレーム間リンク（tracking）を、**trackpy を処理エンジン**として、**napari を可視化・対話調整環境**として統合する Python ライブラリおよび napari プラグイン

---

## 1. 目的

### 1.1 目的

* 顕微鏡で取得した **三次元時系列データ（t, z, y, x）**から、粒子候補を検出し、フレーム間で同一粒子をリンクして **軌跡（track）**を生成する。
* 粒子形状が不定形・ノイジーであり、**ハイパーパラメータを手動で調整**しながら品質を上げられる設計とする。
* 研究用途（再現性・監査性重視）として、**パラメータ・入出力・環境情報**を記録し、再実行可能なパイプラインを提供する。

### 1.2 成果物

1. Python ライブラリ（例：`pt3d` など）
2. napari プラグイン（GUI）

   * 画像表示＋検出点（Points）＋軌跡（Tracks）の重ね合わせ
   * パラメータ変更 → 部分時間範囲で試行 → 結果を即時確認
3. CLI（任意。ただし研究現場でのバッチ処理を想定し推奨）

---

## 2. 想定ユーザー・ユースケース

* 顕微鏡研究者（2Dでなく3D+tが主対象）
* 用途例

  * 単粒子追跡、微粒子の輸送解析、流体中粒子の動態解析
  * 粒子の出現/消失や一時的な検出欠損があるデータ

---

## 3. スコープ

### 3.1 スコープ内

* 3D+t ボリュームスタック（`(t, z, y, x)`）の読み込み・正規化・前処理
* trackpy による N次元検出（locate/batch 相当の呼び出し）
* trackpy によるリンク（`link_df` 相当）
* 追跡結果の QC（短い軌跡除外、外れ値除外、統計量計算）
* napari 表示（Image / Points / Tracks）
* 設定ファイル（YAML/JSON）によるパラメータ管理と、実行結果への埋め込み（プロビナンス）

### 3.2 スコープ外（v0.1では実装しない）

* 深層学習ベースのセグメンテーション（StarDist / Cellpose 等）の学習・推論自体

  * ただし、将来拡張として「検出器バックエンド差し替え」は設計に含める
* マルチオブジェクトの分裂/融合イベントの厳密推定（u-track 的な高度イベントモデル）
* 大規模クラスタ分散実行（Dask cluster 等）

---

## 4. 用語定義

* **Volume**：1時刻の 3D 配列（`(z, y, x)`）
* **Frame**：時刻 `t` の Volume（`t` index）
* **Detection**：各 frame 内で粒子候補の位置（必要に応じて特徴量）を推定する工程
* **Linking/Tracking**：連続フレーム間で同一粒子を対応付けし、track ID を付与する工程
* **Track**：`track_id` に紐づく時系列点列
* **Anisotropy**：z ピッチと xy ピクセルサイズが異なる（z が粗い等）こと

---

## 5. 設計方針

1. **検出と追跡の分離（track-by-detection）**

   * 検出が不安定なままリンクを頑張らない
   * GUI でも検出・リンクを別ボタンに分ける

2. **可視化は napari、計算は trackpy**

   * napari：人間の目で誤検出・誤リンクを判定しやすい重ね表示を提供
   * trackpy：検出・リンク計算と DataFrame 出力を担う

3. **3D異方性を前提**

   * z方向の解像度差を扱えるよう、距離計算やスケーリング手段を仕様化する

4. **再現性（Reproducibility）**

   * 入力データの識別子（パス、ハッシュ、サイズ、軸順）
   * パラメータ（検出・追跡・前処理・スケーリング）
   * バージョン情報（Python、依存パッケージ）
     を出力に記録する。

---

## 6. 全体アーキテクチャ

### 6.1 コンポーネント

* `io`：入力読み込み、軸正規化、メタデータ抽出、出力保存
* `preprocess`：フィルタ、背景補正、正規化、マスク適用
* `detect`：trackpy を用いた 3D 検出（N次元 locate/batch 呼び出し）
* `track`：trackpy を用いたリンク（link_df）＋異方性補正
* `postprocess`：軌跡フィルタ、特徴量計算、QC統計
* `viz_napari`：napari へのレイヤ投入、更新、GUIウィジェット
* `config`：設定スキーマ（Pydantic 等）、入出力、バリデーション
* `pipeline`：上記のオーケストレーション（実行順序と成果物の統合）
* `cli`：設定ファイルでのバッチ実行（任意だが推奨）

### 6.2 データフロー（標準）

1. Load（IO）
2. Preprocess（任意）
3. Detect（per-frame）
4. Track（link）
5. Postprocess/QC
6. Export（CSV/Parquet + 設定 + ログ）
7. Visualize（napari）

---

## 7. データモデル（内部標準）

### 7.1 軸順の標準化

* 内部標準：`(t, z, y, x)`（time-first）
* 入力が異なる場合は、IO層で必ず内部標準に変換する。

### 7.2 物理スケール（任意だが推奨）

* `voxel_size_um = (z_um, y_um, x_um)`
* 追跡距離や速度を物理単位で扱う場合、内部で座標スケーリングを行えること。

### 7.3 検出出力（DataFrame）

必須カラム：

* `frame`（int）
* `z`, `y`, `x`（float：座標。サブピクセル可）

推奨カラム（trackpy が出す/使うことが多い）：

* `mass`, `size`, `signal`, `raw_mass` 等（利用できる場合）
* `quality`（後段フィルタの統一概念。ない場合は `mass` 等で代替）

### 7.4 追跡出力（DataFrame）

上記＋必須：

* `particle`（int：track_id）

### 7.5 napari Tracks 形式

* napari の Tracks は `N x (D+1)` の数値配列で表現する（例：`[track_id, t, z, y, x]`）。
* 本ライブラリは `DataFrame -> tracks_array` 変換関数を提供する。

---

## 8. 入出力仕様

### 8.1 入力サポート

必須（v0.1）：

* `numpy.ndarray`（既に `(t, z, y, x)` になっているもの）
* `zarr`（ローカル。将来 OME-Zarr も想定）

推奨（可能なら v0.1に含める）：

* `tifffile` での TIFF/OME-TIFF 読み込み

### 8.2 出力

必須：

* `detections.parquet`（または `csv`）
* `tracks.parquet`（または `csv`）
* `config.yaml`（実行に用いた設定の完全コピー）
* `run.json`（プロビナンス：入力情報・バージョン・日時・実行時間・例外等）

任意：

* napari 用セッション再現（レイヤ保存）機能（v0.2以降でもよい）

---

## 9. 設定（Config）仕様

### 9.1 形式

* YAML（推奨）または JSON
* `Pydantic` 等で schema validation を行う

### 9.2 設定カテゴリ

1. `input`

   * path / dataset key / axis order / dtype
2. `preprocess`

   * 背景補正、平滑化、正規化、マスク、閾値処理
3. `detect`

   * trackpy locate/batch 相当パラメータ（3D対応）
4. `track`

   * link_df 相当パラメータ
   * 異方性補正（座標スケーリング）設定
5. `postprocess`

   * stub 除去（最短長）
   * 外れ値速度除外
6. `export`

   * 出力形式、パス、上書き可否
7. `napari`

   * レイヤ名、表示設定、サブセット範囲

---

## 10. 検出仕様（Detection）

### 10.1 検出エンジン

* v0.1：trackpy を用いた N次元検出を第一選択とする。
* 入力：`volume[z,y,x]`（単一 frame）または `frames`（複数 frame）
* 出力：DataFrame（`frame,z,y,x,...`）

### 10.2 主要パラメータ（3D）

* `diameter = (dz, dy, dx)`

  * **奇数**であること（検証してエラー）
  * z の解像度が粗い場合は `dz` を小さめにするなど、異方性を意識して指定可能にする
* `minmass`：ノイズ除去の主要ノブ
* `threshold`：背景が不安定な場合の補助ノブ
* `separation`：近接粒子の分離条件
* `invert`：暗い粒子の場合
* `preprocess`：trackpy 内蔵前処理を使う/使わない

### 10.3 実行単位

* `detect(frame_index=t)`：単一フレームの検出
* `detect(range=t0:t1)`：時間範囲の検出（napari での対話用途）
* `detect_all()`：全フレーム検出（バッチ用途）

### 10.4 エッジケース対応

* 検出点数が 0 のフレームを許容（空DataFrameで返す）
* 破綻するフレーム（例外）時は

  * 標準：例外を上げる
  * オプション：ログに記録し、そのフレームをスキップして継続

---

## 11. 追跡仕様（Linking/Tracking）

### 11.1 追跡エンジン

* v0.1：trackpy の `link_df` を使用（DataFrame に `particle` を付与）

### 11.2 主要パラメータ

* `search_range`：フレーム間最大移動距離（最重要）
* `memory`：検出欠損を許容するフレーム数
* `adaptive_stop`, `adaptive_step`：密な場でのサブネット過大問題の緩和（任意）

### 11.3 異方性（zスケール差）対策

本ライブラリは以下の2方式をサポートする。

* **方式A：座標スケーリング（推奨）**

  * リンク前に座標を `z' = z * (z_um / x_um)` 等でスケールし、距離計算を等方化する。
  * 出力時は元座標（未スケール）も保持する（例：`z_raw,y_raw,x_raw` を保持）。
* **方式B：事前にボリュームを等方リサンプリング**

  * `preprocess` 側でリサンプルしてから検出・追跡する（計算コスト増、画質劣化可能性）

v0.1 では方式Aを必須搭載、方式Bは任意搭載。

### 11.4 Track IDの連番規則

* `particle`（track_id）は trackpy の出力に準拠し、0以上の整数とする。
* 出力の安定化のため、必要なら `relabel_tracks(mode="dense")` を提供（オプション）。

---

## 12. 後処理（Postprocess/QC）

### 12.1 最小要件（v0.1）

* stub 除去：短すぎる軌跡の除外（`min_track_length`）
* 統計：trackごとの長さ、平均速度、最大速度（物理単位があれば µm/s も）

### 12.2 任意（v0.2候補）

* ドリフト推定・補正
* 局所密度や MSD（mean squared displacement）計算

---

## 13. napari プラグイン仕様（GUI）

### 13.1 目的

* 画像に対して検出点・軌跡を重ね、誤検出/誤リンクを**視覚的に**確認しながらパラメータを調整できるようにする。

### 13.2 レイヤ構成

* `Image`: 4D (t,z,y,x)
* `Points`: 検出点（座標は `[t,z,y,x]` で表示できる形に整形）
* `Tracks`: 軌跡（配列 `[track_id, t, z, y, x]`）

### 13.3 UI（最小セット）

1. 入力/サブセット

   * 現在フレーム `t` のみ / 範囲 `t0..t1` の指定
2. Detection パネル

   * `diameter (dz,dy,dx)`、`minmass`、`threshold`、`separation`、`invert`
   * 実行ボタン：`Run Detection (subset)`
3. Tracking パネル

   * `search_range`、`memory`、（任意）`adaptive_stop/adaptive_step`
   * 実行ボタン：`Run Tracking (on current detections)`
4. Export パネル

   * `Export detections/tracks/config/run` の保存先
5. ログ表示

   * 検出点数、track数、平均長、例外メッセージ等

### 13.4 実行方式（重要）

* napari UI を固めないために、重い処理は **バックグラウンド worker**（napari推奨のスレッドワーカー等）で実行する。
* 実行中は UI 上で状態（running / done / error）を明示する。

### 13.5 キャッシュ

* 同一サブセット・同一パラメータでの再実行を避けるため、

  * `detections_cache[(t_range, detect_config_hash)] = df`
  * `tracks_cache[(t_range, detect_hash, track_hash)] = df`
    を保持可能にする（メモリ制限に注意）。

---

## 14. Python API 仕様（ライブラリ）

### 14.1 公開API（案）

#### Config

* `PipelineConfig`（入力・前処理・検出・追跡・後処理・出力）
* `DetectionConfig`
* `TrackingConfig`
* `PreprocessConfig`
* `ExportConfig`

#### Pipeline 実行

* `run_pipeline(config: PipelineConfig) -> PipelineResult`

  * `PipelineResult` は `detections_df`, `tracks_df`, `run_info` を保持

#### 個別実行

* `load_volume_series(...) -> np.ndarray`（返り値は `(t,z,y,x)`）
* `preprocess(series, config) -> series`
* `detect(series or subset, detect_config) -> detections_df`
* `track(detections_df, tracking_config, voxel_size_um=None) -> tracks_df`
* `postprocess(tracks_df, post_config) -> tracks_df`

#### napari 変換

* `to_napari_points(detections_df) -> np.ndarray`
* `to_napari_tracks(tracks_df) -> np.ndarray`

#### エクスポート

* `export_results(result, export_config)`

### 14.2 例外設計

* `ConfigError`：設定不正（diameter偶数、軸順不正など）
* `DataError`：入力データ不正（次元不一致など）
* `ProcessingError`：検出/追跡失敗（内部例外をラップ）

---

## 15. 非機能要件

### 15.1 性能

* 最初は全フレーム一括ではなく、**サブセット処理**が高速に回ること（napari調整用途）
* 大規模データに備えて、将来 `dask array` 入力も視野（v0.2）

### 15.2 再現性・監査

* 実行時に以下を `run.json` に保存

  * 入力ファイル情報（パス、サイズ、mtime、任意でhash）
  * config 全内容
  * 依存パッケージバージョン（pip freeze相当の要約）
  * 実行開始/終了時刻、処理時間
  * 例外発生時はトレース要約

### 15.3 可搬性

* OS：Windows/macOS/Linux
* Python：3.10+ を想定（プロジェクトで決定）

---

## 16. テスト要件

### 16.1 単体テスト

* Config validation（diameter 奇数、軸順変換）
* `DataFrame -> napari` 変換の正しさ
* 空入力、空フレーム時の挙動

### 16.2 結合テスト

* 合成データ（既知の粒子位置）で検出精度・追跡精度が一定以上
* 異方性（z_um != x_um）ケースの距離スケーリングが妥当

### 16.3 回帰テスト

* 代表データセット（小）に対し、出力 track 数や平均長などの統計が大きく崩れないこと

---

## 17. 依存関係（推奨）

必須：

* numpy, pandas
* trackpy
* napari（プラグイン側）
* pydantic（config validation）
* pyyaml（YAML）

推奨：

* tifffile（TIFF/OME-TIFF）
* zarr（大規模データ）
* dask（将来）
* scikit-image（前処理）

---

## 18. リポジトリ構成（案）

```
pt3d/
  pyproject.toml
  src/pt3d/
    __init__.py
    config.py
    io.py
    preprocess.py
    detect.py
    track.py
    postprocess.py
    napari/
      __init__.py
      plugin.py
      widgets.py
      layers.py
    pipeline.py
    export.py
    utils.py
  tests/
    test_config.py
    test_convert_napari.py
    test_pipeline_synth.py
  docs/
    spec.md   <-- 本仕様書
    usage.md
  examples/
    notebook_2d_to_3d.ipynb
    run_cli_example.yaml
```

---

## 19. 実装優先順位（MVPロードマップ）

### v0.1（MVP）

* `(t,z,y,x)` ndarray 入力
* trackpy による 3D検出 → link_df
* 座標スケーリング方式による異方性対応
* napari 表示（Image/Points/Tracks）
* YAML config + export（tracks/detections/run）

### v0.2

* zarr/OME-Zarr 入力
* キャッシュ強化、dask対応の布石
* QC指標の拡充（MSD等）
* 検出器バックエンド差し替えインターフェース

---

## 20. 受け入れ基準（Acceptance Criteria）

* 3D+t データを読み込み、指定したパラメータで

  1. detections が DataFrame として得られる
  2. tracks が `particle` 付き DataFrame として得られる
  3. napari 上で Image + Points + Tracks が同時表示できる
  4. config と run 情報が出力され、同一入力・同一設定で再実行した場合に同等結果が得られる
* 異方性設定（例：`voxel_size_um=(0.8, 0.2, 0.2)`）で、z方向の距離が適切に補正される。

---

## 補足：設計上の「重要な落とし穴」と対策（仕様に含めるべき事項）

1. **軸順の混乱**：napariは多次元を柔軟に扱えるが、trackpy側の DataFrame 座標列と一致しないと即破綻する
   → 仕様として「内部標準 `(t,z,y,x)`」「DataFrame カラム `frame,z,y,x`」を固定する。

2. **異方性を無視した距離**：zが粗いのに (z,y,x) を同じ距離で扱うと誤リンクが増える
   → 仕様として「座標スケーリング方式」を v0.1 で必須。

3. **検出が不安定なままリンク**：リンクパラメータ調整で泥沼化
   → GUIは検出と追跡を分離し、検出を先に確定する導線にする。

---

# 次にやると実装が早いこと（仕様の確定ポイント）

仕様を最終確定するうえで、実装に影響が大きい“決め”は次の3点です（質問というより決定事項の候補です。あなたの判断で埋めてください）。

1. **入力フォーマットの最優先**：OME-TIFFを必須にするか、まずは ndarray/zarr のみで始めるか
2. **距離単位**：

   * 追跡パラメータ `search_range` を「ピクセル基準」で統一するか
   * 「µm基準」を標準にするか（その場合 voxel_size_um を必須入力にする）
3. **前処理の責務**：

   * v0.1でどこまで（背景補正、DoG/LoG、正規化）を標準提供するか
   * もしくはユーザーが前処理済みデータを渡す前提にするか

---

必要なら、この仕様書の内容を前提に、Codex に渡すための「タスク分割（issueテンプレ）」「各モジュールの関数シグネチャ一覧」「設定YAMLの具体例」まで落として提示します。
