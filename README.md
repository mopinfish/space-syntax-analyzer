# Space Syntax解析システム

OpenStreetMapデータを活用したSpace Syntax理論に基づく都市空間構造の定量解析システム

## 🌟 概要

本システムは、ビル・ヒリアー（Bill Hillier）によって開発されたSpace Syntax理論に基づき、都市空間の構造特性を定量的に分析するPythonアプリケーションです。OpenStreetMapの豊富な道路ネットワークデータを活用し、以下の主要指標を計算・可視化します：

- **Integration（統合性）**: 空間のアクセシビリティと中心性
- **Connectivity（接続性）**: 直接接続数と交通結節点特性  
- **Choice（選択性）**: 経路選択における重要度
- **Depth（深度）**: システム内での相対的位置
- **Intelligibility（理解容易性）**: 空間認知の容易さ

## 🎯 主要機能

### 📊 解析機能
- **Axial Map解析**: 道路軸線レベルでの大域的分析
- **Segment Map解析**: 角度重み付きセグメント分析
- **Local/Global解析**: 限定半径と全域での多層分析
- **統計分析**: 基本統計量・分布・相関分析

### 🗺️ 可視化機能
- **指標別色分け地図**: 各Space Syntax指標の空間分布
- **統計グラフ**: ヒストグラム・散布図・箱ひげ図
- **相関分析**: ヒートマップと散布図行列
- **インタラクティブマップ**: Webブラウザ対応の動的地図

### 📋 レポート機能
- **自動PDF生成**: 包括的な分析レポート
- **多言語対応**: 日本語・英語
- **カスタマイズ**: テンプレート編集可能

## 🚀 クイックスタート

### 1. インストール

```bash
# リポジトリクローン
git clone https://github.com/your-org/space-syntax-analyzer.git
cd space-syntax-analyzer

# 依存関係インストール  
pip install -r requirements.txt

# パッケージインストール（オプション）
pip install -e .
```

### 2. 基本使用例

```bash
# 地名指定での解析
python main.py --place "Shibuya, Tokyo, Japan"

# 境界座標での解析
python main.py --bbox 35.6580,139.6956,35.6650,139.7056

# 詳細設定での解析
python main.py --place "Kyoto, Japan" --config config/detailed_analysis.json --analysis-type both
```

### 3. 出力確認

```bash
# 出力ディレクトリ確認
ls output/
# → axial/, segment/, *.pdf, *.csv, *.html が生成される
```

## 📋 システム要件

### 必須環境
- **Python**: 3.8以上
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **メモリ**: 4GB以上推奨（大規模解析時は8GB+）
- **ストレージ**: 1GB以上の空き容量

### 必須依存関係
- NetworkX 2.5+ (グラフ解析)
- OSMnx 1.0+ (OpenStreetMapデータ)
- GeoPandas 0.10+ (地理空間データ)
- Matplotlib 3.5+ (可視化)
- SciPy 1.8+ (統計計算)

### オプション依存関係
- Folium 0.12+ (インタラクティブマップ)
- WeasyPrint 54+ (PDF生成)
- Plotly 5.0+ (高度な可視化)

## 📖 使用方法

### コマンドライン引数

```bash
python main.py [OPTIONS]

オプション:
  --place TEXT          解析対象地域名 (例: "Tokyo, Japan")
  --bbox TEXT           境界座標 (south,west,north,east)
  --admin TEXT          行政区域名
  --config TEXT         設定ファイルパス
  --analysis-type TEXT  解析タイプ [axial|segment|both]
  --output-dir TEXT     出力ディレクトリ
  --log-level TEXT      ログレベル [DEBUG|INFO|WARNING|ERROR]
  --quiet              静音モード
  --help               ヘルプ表示
```

### 設定ファイル

```json
{
  "osm_settings": {
    "network_type": "walk",    // walk, drive, bike, all
    "simplify": true,
    "timeout": 180
  },
  "analysis_settings": {
    "local_radii": [400, 800, 1200],
    "calculate_global": true,
    "normalize": true
  },
  "visualization_settings": {
    "color_map": "viridis",
    "dpi": 300,
    "save_formats": ["png", "svg"]
  }
}
```

### プログラマティック使用

```python
from src.osm_data_loader import OSMDataLoader
from src.space_syntax_analyzer import SpaceSyntaxAnalyzer
from src.visualization import Visualizer

# データ取得
loader = OSMDataLoader(settings)
graph = loader.load_by_place("Shibuya, Tokyo, Japan")

# 解析実行
analyzer = SpaceSyntaxAnalyzer(settings)
results = analyzer.analyze_axial_map(graph)

# 可視化
visualizer = Visualizer(settings)
visualizer.create_map_visualizations(
    results['graph'], 
    results['metrics'], 
    output_dir
)
```

## 📊 出力ファイル構成

```
output/
├── axial/                    # Axial Map解析結果
│   ├── map_integration.png   # Integration分布図
│   ├── map_connectivity.png  # Connectivity分布図
│   ├── map_choice.png        # Choice分布図
│   └── statistics_table.csv  # 統計表
├── segment/                  # Segment Map解析結果
│   ├── map_angular_integration.png
│   ├── map_angular_choice.png
│   └── correlation_heatmap.png
├── space_syntax_report.pdf  # 総合分析レポート
├── interactive_map.html     # インタラクティブマップ
└── analysis_results.json    # 生データ（JSON形式）
```

## 🔧 カスタマイズ

### 新しい指標の追加

```python
# src/space_syntax_analyzer.py
def calculate_custom_metric(self, graph):
    """カスタム指標の計算"""
    custom_values = {}
    for node in graph.nodes():
        # カスタム計算ロジック
        custom_values[node] = calculate_value(node)
    return pd.Series(custom_values)
```

### 可視化スタイルの変更

```python
# src/visualization.py  
def _get_custom_colormap(self):
    """カスタムカラーマップ"""
    return 'plasma'  # matplotlib colormap name
```

### レポートテンプレートの編集

```python
# src/report_generator.py
def _get_css_style(self):
    """カスタムCSS定義"""
    return """
    .custom-section {
        background-color: #f0f8ff;
        border-left: 4px solid #4169e1;
    }
    """
```

## 🔬 解析事例

### 都市中心部の分析
```bash
# 東京・渋谷駅周辺
python main.py --place "Shibuya Station, Tokyo" --analysis-type both

# 大阪・梅田地区  
python main.py --place "Umeda, Osaka" --config config/urban_analysis.json
```

### 住宅地域の分析
```bash
# 世田谷区の住宅街
python main.py --place "Setagaya, Tokyo" --analysis-type segment
```

### 歴史的市街地の分析
```bash
# 京都・祇園地区
python main.py --place "Gion, Kyoto" --config config/detailed_analysis.json
```

## 📚 理論的背景

### Space Syntax理論とは

Space Syntaxは、建築・都市空間の形態的特性と社会的活動の関係を定量的に分析する理論です：

1. **空間は社会を形成する**: 物理的空間の構造が人々の行動パターンに影響
2. **構成（Configuration）**: 部分と全体の関係性が重要
3. **自然的動線（Natural Movement）**: 空間構造から予測可能な移動パターン

### 主要指標の解釈

| 指標 | 意味 | 高い値の特徴 | 都市計画への示唆 |
|------|------|-------------|-----------------|
| **Integration** | 全体からのアクセス性 | 中心性が高く、人が集まりやすい | 商業・業務地区に適している |
| **Connectivity** | 直接接続数 | 交通の結節点、選択肢が豊富 | 交通インフラの重要地点 |
| **Choice** | 経路選択での重要度 | 通過交通が多い | 商業立地、交通量対策が必要 |
| **Depth** | システム内での深さ | 中心からの距離が近い | アクセシビリティの指標 |
| **Intelligibility** | 理解容易性 | 方向感覚を失いにくい | サイン計画、ランドマーク配置 |

### Angular Analysis（角度解析）

セグメント解析では、実際の移動における角度変化を考慮：
- **直進性の重視**: 人は曲がり角を嫌う傾向
- **認知負荷**: 角度変化が少ないほど移動しやすい
- **実測値との相関**: 実際の歩行者数との相関が高い

## 🔍 結果の解釈ガイド

### Integration値の読み方
- **高Integration地域**: 都市の中心核、商業集積に適
- **低Integration地域**: 住宅地、静穏な環境
- **空間格差**: 値の分散が大きい場合は不平等な空間構造

### Intelligibility分析
- **r > 0.7**: 非常に理解しやすい空間構造
- **0.4 < r < 0.7**: まずまず理解しやすい
- **r < 0.4**: 複雑で迷いやすい構造

### 相関分析の活用
- **Integration-Connectivity相関**: 都市構造の整合性
- **Choice-Integration相関**: 通過性と到達性のバランス
- **局所-大域相関**: 多層的な空間構造の理解

## 🛠️ 開発・拡張

### 開発環境セットアップ

```bash
# 開発用依存関係インストール
pip install -e ".[dev]"

# テスト実行
pytest tests/

# コード品質チェック
black src/ tests/
flake8 src/ tests/
mypy src/
```

### テストデータ

```bash
# 小規模テストデータでの動作確認
python main.py --place "Shibuya Crossing, Tokyo" --analysis-type axial
```

### 新機能の追加

1. **新しい指標**: `src/space_syntax_analyzer.py`に追加
2. **新しい可視化**: `src/visualization.py`に追加  
3. **新しい出力形式**: `src/report_generator.py`に追加

### API拡張

```python
# 将来的なWeb API対応例
from flask import Flask, request, jsonify
from src.space_syntax_analyzer import SpaceSyntaxAnalyzer

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
def analyze_location():
    data = request.json
    # 解析処理
    results = analyzer.analyze(data['location'])
    return jsonify(results)
```

## 📈 パフォーマンス最適化

### 大規模データ対応

```python
# メモリ効率的な処理
config['osm_settings']['max_query_area_size'] = 25000000  # より小さく
config['analysis_settings']['calculate_local'] = False    # 局所解析省略
```

### 並列処理

```python
# マルチプロセッシング対応
from multiprocessing import Pool

def analyze_city(city_name):
    # 都市別解析処理
    pass

cities = ["Tokyo", "Osaka", "Kyoto", "Yokohama"]
with Pool(4) as p:
    results = p.map(analyze_city, cities)
```

### キャッシュ活用

```python
# OSMnxキャッシュ設定
import osmnx as ox
ox.settings.use_cache = True
ox.settings.cache_folder = './cache'
```

## 📊 ベンチマーク

| 地域規模 | ノード数 | 処理時間 | メモリ使用量 |
|----------|----------|----------|-------------|
| 小地区 | ~1,000 | 30秒 | 500MB |
| 中地区 | ~5,000 | 2分 | 1.5GB |
| 大地区 | ~20,000 | 10分 | 4GB |
| 都市レベル | ~50,000 | 30分 | 8GB+ |

## 🔧 トラブルシューティング

### よくある問題と解決法

**1. メモリ不足エラー**
```bash
# 解決策: より小さな領域で解析
python main.py --bbox 35.6580,139.6956,35.6620,139.7000  # 範囲を縮小
```

**2. ネットワークタイムアウト**
```json
// config.jsonで設定
{
  "osm_settings": {
    "timeout": 300  // 5分に延長
  }
}
```

**3. 座標系エラー**
```python
# デバッグ用: 座標系確認
import geopandas as gpd
gdf = gpd.read_file("your_data.geojson")
print(gdf.crs)  # 座標系確認
```

**4. フォント表示問題**
```python
# 日本語フォント設定
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic']
```

**5. PDF生成エラー**
```bash
# WeasyPrint代替
pip uninstall weasyprint
pip install pdfkit
# 注意: wkhtmltopdfの別途インストールが必要
```

### ログ確認

```bash
# 詳細ログでの実行
python main.py --place "Tokyo" --log-level DEBUG

# ログファイル確認
tail -f logs/space_syntax_$(date +%Y%m%d).log
```

### システム情報確認

```python
# システム情報の表示
python -c "from src.utils import print_system_info; print_system_info()"
```

## 🤝 コミュニティ・サポート

### 貢献方法

1. **Issue報告**: バグや機能要求の報告
2. **Pull Request**: コード改善の提案
3. **ドキュメント**: 使用例や解説の追加
4. **テストケース**: 新しいテストの追加

### ガイドライン

- コードスタイル: Black + flake8準拠
- テスト: pytest使用、カバレッジ80%以上
- ドキュメント: docstring必須
- 言語: 日本語・英語両対応

### 連絡先

- **GitHub Issues**: 技術的な問題・機能要求
- **Discussions**: 一般的な質問・使用例の共有
- **Email**: 重要な問題や商業利用相談

## 📜 ライセンス・引用

### ライセンス
MIT License - 学術・商用利用可能

### 引用方法

学術論文での引用例：
```
Space Syntax Analysis System. (2025). 
OpenStreetMapデータを活用したSpace Syntax解析システム. 
Retrieved from https://github.com/your-org/space-syntax-analyzer
```

BibTeX:
```bibtex
@software{space_syntax_analyzer_2025,
  title = {Space Syntax Analysis System},
  author = {Space Syntax Development Team},
  year = {2025},
  url = {https://github.com/your-org/space-syntax-analyzer},
  version = {1.0.0}
}
```

### 参考文献

1. Hillier, B., & Hanson, J. (1984). *The Social Logic of Space*. Cambridge University Press.
2. Hillier, B. (2007). *Space is the Machine*. Space Syntax Limited.
3. Al-Sayed, K., Turner, A., Hillier, B., Iida, S., & Penn, A. (2014). *Space Syntax Methodology*. Bartlett School of Architecture, UCL.
4. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.

## 🔄 更新履歴

### v0.0.1 (2025-01)
- 初回リリース
- Axial/Segment Map解析対応
- 基本的な可視化機能
- PDF レポート生成

### 今後の予定

**v1.0.0**
- WebAPI対応
- リアルタイム解析
- 機械学習機能統合

**v1.1.0**  
- 時系列分析
- 複数都市比較機能
- クラウド対応

**v2.0.0**
- 3D空間解析
- VR/AR可視化
- AIによる自動解釈

---

## 🌟 謝辞

本システムの開発にあたり、以下の方々・組織に感謝いたします：

- Space Syntax理論の創始者 Bill Hillier教授
- OSMnxライブラリ開発者 Geoff Boeing博士  
- OpenStreetMapコミュニティ
- Python地理空間解析コミュニティ
- ベータテスター・フィードバック提供者の皆様

Space Syntaxを通じた都市空間理解の発展に貢献できれば幸いです。

---

**🚀 今すぐ始める:**
```bash
git clone https://github.com/your-org/space-syntax-analyzer.git
cd space-syntax-analyzer
pip install -r requirements.txt
python main.py --place "Your City"
```
