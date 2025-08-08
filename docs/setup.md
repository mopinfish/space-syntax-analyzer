# Space Syntax解析システム ダウンロード・セットアップガイド

## 📁 推奨ディレクトリ構造

以下の構造でディレクトリを作成し、各アーティファクトを配置してください：

```
space_syntax_system/
├── main.py                          # ←メインエントリーポイント
├── README.md                        # ←プロジェクト概要
├── requirements.txt                 # ←Python依存関係
├── setup.py                        # ←パッケージセットアップ
├── config/
│   └── default_config.json         # ←デフォルト設定ファイル
├── src/
│   ├── __init__.py                  # ←パッケージ初期化（空ファイル）
│   ├── config_manager.py           # ←設定管理モジュール
│   ├── osm_data_loader.py          # ←OSMデータローダー
│   ├── graph_builder.py            # ←グラフビルダー
│   ├── space_syntax_analyzer.py    # ←Space Syntax解析エンジン
│   ├── visualization.py            # ←可視化モジュール
│   ├── report_generator.py         # ←レポート生成モジュール
│   └── utils.py                     # ←ユーティリティ
├── docs/
│   ├── requirements.md             # ←要件定義書
│   └── screen_flow.md              # ←画面フロー図
├── output/                         # ←出力ディレクトリ（空で作成）
└── logs/                           # ←ログディレクトリ（空で作成）
```

## 🚀 セットアップ手順

### 1. ディレクトリ作成
```bash
mkdir space_syntax_system
cd space_syntax_system
mkdir config src docs output logs
```

### 2. ファイルダウンロード

以下のアーティファクトをそれぞれダウンロードして配置：

#### 📋 ドキュメント
- **要件定義書** → `docs/requirements.md`
- **画面フロー図** → `docs/screen_flow.md`
- **README.md** → `README.md`

#### 💻 ソースコード
- **main.py** → `main.py`
- **config_manager.py** → `src/config_manager.py`
- **osm_data_loader.py** → `src/osm_data_loader.py`
- **graph_builder.py** → `src/graph_builder.py`
- **space_syntax_analyzer.py** → `src/space_syntax_analyzer.py`
- **visualization.py** → `src/visualization.py`
- **report_generator.py** → `src/report_generator.py`
- **utils.py** → `src/utils.py`

#### ⚙️ 設定・環境
- **default_config.json** → `config/default_config.json`
- **requirements.txt** → `requirements.txt`
- **setup.py** → `setup.py`

### 3. 初期化ファイル作成
```bash
# __init__.pyファイルの作成
echo '"""Space Syntax Analysis System Package"""' > src/__init__.py
echo '__version__ = "1.0.0"' >> src/__init__.py
```

### 4. 依存関係インストール
```bash
pip install -r requirements.txt
```

### 5. 動作確認
```bash
python main.py --help
```

## 📝 各ファイルの概要

| ファイル | 機能 | 重要度 |
|----------|------|--------|
| **main.py** | システムメイン制御 | ⭐⭐⭐ |
| **config_manager.py** | 設定管理 | ⭐⭐⭐ |
| **osm_data_loader.py** | OSMデータ取得 | ⭐⭐⭐ |
| **graph_builder.py** | グラフ構築 | ⭐⭐⭐ |
| **space_syntax_analyzer.py** | 解析エンジン | ⭐⭐⭐ |
| **visualization.py** | 可視化 | ⭐⭐ |
| **report_generator.py** | レポート生成 | ⭐⭐ |
| **utils.py** | ユーティリティ | ⭐⭐ |
| **default_config.json** | デフォルト設定 | ⭐⭐⭐ |
| **requirements.txt** | 依存関係定義 | ⭐⭐⭐ |

## 🎯 最小構成での動作確認

最優先で以下のファイルをダウンロード・配置してください：

1. `main.py` 
2. `src/config_manager.py`
3. `src/osm_data_loader.py`
4. `src/graph_builder.py` 
5. `src/space_syntax_analyzer.py`
6. `src/utils.py`
7. `config/default_config.json`
8. `requirements.txt`
9. `src/__init__.py`（空ファイル）

これらがあれば基本的な解析が実行できます。

## 🔧 トラブルシューティング

### インポートエラーが発生する場合
```bash
# PYTHONPATHの設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# または実行時に指定
python -m main --place "Tokyo, Japan"
```

### 依存関係エラーが発生する場合
```bash
# 個別インストール
pip install networkx osmnx geopandas pandas numpy scipy matplotlib seaborn

# 仮想環境の使用
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 使用例

セットアップ完了後、以下で動作確認：

```bash
# 基本分析
python main.py --place "Shibuya, Tokyo, Japan"

# 詳細分析  
python main.py --place "Kyoto, Japan" --analysis-type both

# 境界座標指定
python main.py --bbox 35.6580,139.6956,35.6650,139.7056
```

## 💡 最適化のヒント

### パフォーマンス向上
- 小さな地域から開始
- `--analysis-type axial` で高速化
- メモリ使用量監視

### 結果の活用
- `output/` ディレクトリの画像確認
- PDF レポートの詳細分析
- CSVデータの外部ツール活用

## 🤝 サポート

- **基本的な使用方法**: README.md参照
- **詳細な解析手法**: docs/requirements.md参照
- **技術的問題**: ログファイル確認（`logs/` ディレクトリ）

---

このガイドに従ってセットアップを行い、Space Syntax解析をお楽しみください！
