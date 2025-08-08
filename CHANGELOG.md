# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- 高度な可視化機能（ヒートマップ、3D表示）
- 複数地域の一括比較分析機能
- 時系列分析機能
- 歩行者流動データとの統合分析
- Webアプリケーション版の開発

## [0.1.0] - 2025-08-08

### Added
- 初回リリース
- スペースシンタックス理論に基づく基本分析機能
  - 回遊性指標（μ、α、β、γ指数）
  - アクセス性指標（平均最短距離、道路密度、交差点密度）
  - 迂回性指標（平均迂回率）
- OSMnxを活用した道路ネットワーク自動取得機能
- 道路幅員（4m）による主要道路と細街路の分類
- 基本的な可視化機能
  - ネットワーク比較表示
  - 指標比較チャート
  - レーダーチャート
- 多様な出力形式サポート
  - CSV、Excel、JSON形式での分析結果出力
  - GeoJSON、Shapefile、GraphML形式での空間データ出力
- 包括的なテストスイート
- 型ヒント完全対応
- uvによるモダンなパッケージ管理
- PyPI配布対応

### Technical Details
- Python 3.9以上対応
- NetworkX v3.0以上対応
- OSMnx v1.9.0以上対応
- 完全な型ヒント実装
- pytest基盤のテストフレームワーク
- Black + isort + ruff による自動コードフォーマット
- mypy による静的型検査

### Documentation
- 包括的なREADME（日本語・英語）
- API リファレンス
- 使用例とサンプルコード
- 開発者向けドキュメント

### Known Limitations
- 大規模ネットワーク（10万ノード超）での性能制限
- OpenStreetMapデータの品質に依存
- インターネット接続が必要（ネットワーク取得時）
- 一部の高度なスペースシンタックス指標は未実装
