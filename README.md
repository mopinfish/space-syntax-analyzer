# space-syntax-analyzer

[![PyPI version](https://badge.fury.io/py/space-syntax-analyzer.svg)](https://badge.fury.io/py/space-syntax-analyzer)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

スペースシンタックス理論に基づいた都市空間の分析を行う Python ライブラリです。OpenStreetMap の道路ネットワークデータを使用して、都市の空間構造を定量的に分析・可視化します。

## 概要

space-syntax-analyzer は、Bill Hillier らによって開発されたスペースシンタックス理論を基盤とし、以下の機能を提供します：

- **道路ネットワークの自動取得**: OSMnx を活用した OpenStreetMap からのデータ取得
- **スペースシンタックス分析**: 回遊性、アクセス性、迂回性の定量的評価
- **道路幅員による分類**: 主要道路（4m 以上）と細街路の分離分析
- **可視化機能**: ネットワーク構造と分析結果の直感的な表示
- **多様な出力形式**: CSV、Excel、JSON、GeoJSON での結果出力

## インストール

### uv を使用する場合（推奨）

```bash
uv add space-syntax-analyzer
```

### pip を使用する場合

```bash
pip install space-syntax-analyzer
```

### 開発版のインストール

```bash
git clone https://github.com/space-syntax/space-syntax-analyzer.git
cd space-syntax-analyzer
uv sync --dev
```

## クイックスタート

### 基本的な使用方法

```python
from space_syntax_analyzer import SpaceSyntaxAnalyzer

# アナライザーの初期化
analyzer = SpaceSyntaxAnalyzer()

# 渋谷駅周辺の分析
results = analyzer.analyze_place("Shibuya, Tokyo, Japan")

# 結果の表示
print(analyzer.generate_report(results, "渋谷駅周辺"))
```

### ネットワークの可視化

```python
# ネットワークと結果を取得
results, (major_net, full_net) = analyzer.analyze_place(
    "Shibuya, Tokyo, Japan",
    return_networks=True
)

# 可視化
analyzer.visualize(major_net, full_net, results)
```

### カスタム分析範囲

```python
# 座標で範囲指定
bbox = (35.6762, 35.6462, 139.7139, 139.6839)  # (north, south, east, west)
results = analyzer.analyze_place(bbox)

# 中心点から1km四方の範囲
from space_syntax_analyzer.utils import create_bbox_from_center
bbox = create_bbox_from_center(35.6612, 139.6989, distance_km=1.0)
results = analyzer.analyze_place(bbox)
```

## 分析指標

### 回遊性指標（Connectivity Metrics）

| 指標          | 計算式             | 説明                          |
| ------------- | ------------------ | ----------------------------- |
| 回路指数（μ） | e - ν + p          | 循環路の数                    |
| α 指数        | μ / (2ν - 5) × 100 | 回遊性の程度（%）             |
| β 指数        | e / ν              | ノードあたりの平均エッジ数    |
| γ 指数        | e / 3(ν - 2) × 100 | 完全グラフに対する連結度（%） |

### アクセス性指標（Accessibility Metrics）

| 指標               | 計算式                 | 説明                             |
| ------------------ | ---------------------- | -------------------------------- |
| 平均最短距離（Di） | 全ペア間最短距離の平均 | アクセスのしやすさ（m）          |
| 道路密度（Dl）     | L / S                  | 単位面積あたりの道路延長（m/ha） |
| 交差点密度（Dc）   | νc / S                 | 単位面積あたりの交差点数（n/ha） |

### 迂回性指標（Circuity Metrics）

| 指標            | 計算式              | 説明       |
| --------------- | ------------------- | ---------- |
| 平均迂回率（A） | 道路距離 / 直線距離 | 迂回の程度 |

## 詳細な使用例

### 複数地域の比較分析

```python
# 複数地域の分析
locations = [
    "Shibuya, Tokyo, Japan",
    "Shinjuku, Tokyo, Japan",
    "Harajuku, Tokyo, Japan"
]

comparison_results = {}
for location in locations:
    results = analyzer.analyze_place(location)
    comparison_results[location] = results

# 比較表の作成
for location, result in comparison_results.items():
    print(f"\n=== {location} ===")
    print(analyzer.generate_report(result))
```

### 結果のエクスポート

```python
# CSV形式でエクスポート
analyzer.export_results(results, "analysis_results.csv", format_type="csv")

# Excel形式でエクスポート
analyzer.export_results(results, "analysis_results.xlsx", format_type="excel")

# ネットワークのGeoJSONエクスポート
analyzer.network_manager.export_network(
    major_net,
    "network.geojson",
    format_type="geojson"
)
```

### カスタム設定

```python
# 道路幅員の閾値を変更
analyzer = SpaceSyntaxAnalyzer(width_threshold=6.0)

# 歩行者ネットワークで分析
analyzer = SpaceSyntaxAnalyzer(network_type="walk")

# カスタム座標系
analyzer = SpaceSyntaxAnalyzer(crs="EPSG:3857")
```

## API リファレンス

### SpaceSyntaxAnalyzer

メインの分析クラスです。

#### メソッド

- `get_network(location, network_filter)`: ネットワーク取得
- `analyze(major_network, full_network, area_ha)`: 分析実行
- `analyze_place(location, return_networks)`: ワンステップ分析
- `visualize(major_network, full_network, results)`: 可視化
- `export_results(results, output_path, format_type)`: 結果出力
- `generate_report(results, location_name)`: レポート生成

### NetworkManager

ネットワークの取得と管理を行います。

#### メソッド

- `get_network(location, network_filter)`: ネットワーク取得
- `calculate_area_ha(graph)`: 面積計算
- `export_network(graph, output_path, format_type)`: ネットワーク出力

### NetworkVisualizer

可視化機能を提供します。

#### メソッド

- `plot_network_comparison(major_network, full_network, results)`: ネットワーク比較表示
- `plot_metrics_comparison(results)`: 指標比較チャート
- `create_metrics_summary_table(results)`: サマリーテーブル作成

## データ形式

### 分析結果の構造

```python
results = {
    "major_network": {
        "nodes": 150,
        "edges": 180,
        "total_length_m": 12500.0,
        "area_ha": 25.0,
        "mu_index": 31,
        "mu_per_ha": 1.24,
        "alpha_index": 10.3,
        "beta_index": 1.2,
        "gamma_index": 40.5,
        "avg_shortest_path": 450.2,
        "road_density": 500.0,
        "intersection_density": 2.4,
        "avg_circuity": 1.35
    },
    "full_network": {
        # 同様の構造
    }
}
```

## 開発者向け情報

### 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/space-syntax/space-syntax-analyzer.git
cd space-syntax-analyzer

# 依存関係のインストール
uv sync --dev

# テストの実行
uv run pytest

# コードフォーマット
uv run black space_syntax_analyzer/
uv run isort space_syntax_analyzer/

# 型チェック
uv run mypy space_syntax_analyzer/
```

### テストカバレッジ

```bash
uv run pytest --cov=space_syntax_analyzer --cov-report=html
```

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

プロジェクトへの貢献を歓迎します！以下の手順でご協力ください：

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 引用

このライブラリを研究で使用する場合は、以下のように引用してください：

```
Space Syntax Analyzer Team. (2025). space-syntax-analyzer: A Python library for space syntax analysis of urban street networks. https://github.com/mopinfish/space-syntax-analyzer
```

## 参考文献

- Hillier, B., & Hanson, J. (1984). The Social Logic of Space. Cambridge University Press.
- Hillier, B. (1996). Space is the Machine. Cambridge University Press.
- van Nes, A., & Yamu, C. (2021). Introduction to Space Syntax in Urban Studies. Springer.

## サポート

- **Issues**: [GitHub Issues](https://github.com/mopinfish/space-syntax-analyzer/issues)
- **ディスカッション**: [GitHub Discussions](https://github.com/mopinfish/space-syntax-analyzer/discussions)
- **ドキュメント**: [公式ドキュメント(WIP)]()

## 更新履歴

### v0.1.0 (2025-08-08)

- 初回リリース
- 基本的なスペースシンタックス指標の計算機能
- OSMnx を活用した道路ネットワーク取得
- 基本的な可視化機能
- CSV/Excel/JSON 出力機能
