# 駅周辺道路ネットワーク分析デモ

## 📋 概要

このデモは、指定した駅の周辺 800m 圏内の道路ネットワークを取得・分析し、各駅の交通アクセシビリティと道路網の特徴を比較分析するツールです。

## 🎯 主な機能

### 🚉 駅周辺分析

- **分析範囲**: 駅から 800m 圏内（設定可変）
- **ネットワーク取得**: OpenStreetMap から自動取得
- **既存データ利用**: GraphML 形式のネットワークファイル読み込み対応

### 📊 分析項目

- **基本ネットワーク分析**: ノード・エッジ数、密度、接続性
- **軸線分析**: Space Syntax 理論に基づく軸線構造解析
- **統合評価**: アクセシビリティスコア、接続性スコア、総合評価

### 🗺️ 可視化機能

- **背景地図付きネットワーク図**: OpenStreetMap 上にネットワーク表示
- **軸線分析図**: Integration 値による色分け表示
- **駅間比較チャート**: 複数駅の指標比較
- **統合レポート**: Markdown 形式の詳細レポート

## 🚀 使用方法

### 1. 環境準備

```bash
# 必要パッケージのインストール
uv add osmnx networkx pandas matplotlib numpy geopandas contextily japanize-matplotlib

# または pip を使用
pip install osmnx networkx pandas matplotlib numpy geopandas contextily japanize-matplotlib
```

### 2. 設定ファイルの準備

#### 自動生成

```bash
# デモを実行すると設定ファイルを自動生成
python examples/station_analysis_demo.py
```

#### 手動作成

`station_config.json` ファイルを作成：

```json
{
  "analysis_settings": {
    "radius_meters": 800,
    "network_type": "drive",
    "include_analysis": ["basic", "axial", "integration"],
    "save_graphml": true,
    "save_visualization": true,
    "background_map": true
  },
  "output_directory": "station_analysis_output",
  "stations": [
    {
      "id": "shibuya",
      "name": "渋谷駅",
      "location": "Shibuya Station, Tokyo, Japan",
      "coordinates": [35.658, 139.7016],
      "graphml_path": null,
      "description": "若者文化の中心地"
    }
  ]
}
```

### 3. 実行

#### 基本実行

```bash
python examples/station_analysis_demo.py
```

#### スクリプト実行

```bash
chmod +x run_station_analysis.sh
./run_station_analysis.sh
```

## ⚙️ 設定ファイル詳細

### 分析設定 (`analysis_settings`)

| 項目                 | 説明                 | デフォルト値                      |
| -------------------- | -------------------- | --------------------------------- |
| `radius_meters`      | 分析半径（メートル） | 800                               |
| `network_type`       | ネットワーク種別     | "drive"                           |
| `include_analysis`   | 実行する分析項目     | ["basic", "axial", "integration"] |
| `save_graphml`       | GraphML ファイル保存 | true                              |
| `save_visualization` | 可視化ファイル保存   | true                              |
| `background_map`     | 背景地図表示         | true                              |

### 駅設定 (`stations`)

各駅は以下の項目で設定：

| 項目           | 必須 | 説明                        | 例                       |
| -------------- | ---- | --------------------------- | ------------------------ |
| `id`           | ✅   | 駅の一意識別子              | "shibuya"                |
| `name`         | ✅   | 駅名（表示用）              | "渋谷駅"                 |
| `location`     | △    | 地名による指定              | "Shibuya Station, Tokyo" |
| `coordinates`  | △    | 座標による指定 [緯度, 経度] | [35.6580, 139.7016]      |
| `graphml_path` | -    | 既存 GraphML ファイルパス   | "data/shibuya.graphml"   |
| `description`  | -    | 駅の説明                    | "若者文化の中心地"       |

**注意**: `location` または `coordinates` のいずれかは必須です。

## 📁 出力ファイル

分析完了後、`station_analysis_output/` に以下が生成されます：

### 駅別ファイル

- `{駅ID}_network.png` - ネットワーク図（背景地図付き）
- `{駅ID}_axial.png` - 軸線分析図
- `{駅ID}_network.graphml` - ネットワークデータ

### 比較分析ファイル

- `station_comparison.csv` - 駅間比較データ
- `station_comparison_charts.png` - 比較チャート
- `station_analysis_report_{日時}.md` - 統合レポート

## 📊 分析指標の説明

### アクセシビリティスコア

駅周辺の道路網密度と接続性を総合評価（0-100 点）

### 接続性スコア

道路ネットワークの平均次数に基づく接続性評価（0-100 点）

### 総合スコア

アクセシビリティと接続性の平均値（0-100 点）

### 評価レベル

- **A (80-100 点)**: 非常に良好
- **B (65-79 点)**: 良好
- **C (50-64 点)**: 標準
- **D (35-49 点)**: 改善の余地あり
- **E (0-34 点)**: 大幅改善必要

## 🔧 高度な使用方法

### カスタム駅リストの作成

```json
{
  "stations": [
    {
      "id": "custom_station",
      "name": "カスタム駅",
      "coordinates": [35.0, 139.0],
      "description": "独自設定の駅"
    }
  ]
}
```

### 既存ネットワークデータの利用

```json
{
  "id": "existing_data",
  "name": "既存データ駅",
  "graphml_path": "data/existing_network.graphml",
  "description": "事前に取得済みのネットワーク"
}
```

### 分析半径の変更

```json
{
  "analysis_settings": {
    "radius_meters": 1000, // 1kmに変更
    "network_type": "walk" // 歩行者ネットワークに変更
  }
}
```

## 🌍 対応地域

OpenStreetMap のデータが利用可能な全世界の地域で使用できます。

## ⚠️ 注意事項

- インターネット接続が必要（OpenStreetMap データ取得のため）
- 大都市部では処理時間が長くなる場合があります
- GraphML ファイルは分析後の再利用に便利です
- 背景地図機能には contextily と geopandas が必要です

## 🐛 トラブルシューティング

### よくある問題

**Q: "ネットワーク取得エラー"が発生する**
A: 座標や地名が正確か確認してください。または範囲を狭めてみてください。

**Q: 背景地図が表示されない**  
A: `contextily`と`geopandas`がインストールされているか確認してください。

**Q: 日本語が表示されない**
A: `japanize-matplotlib`をインストールしてください。

### ログ確認

```bash
# 詳細ログを表示
python examples/station_analysis_demo.py 2>&1 | tee analysis.log
```

## 📝 ライセンス

このデモは space_syntax_analyzer パッケージの一部として提供されます。
