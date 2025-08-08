"""
Space Syntax Analyzer - 道路ネットワークのSpace Syntax分析ツール

このパッケージはOpenStreetMapからの道路ネットワークデータを使用して、
Space Syntax理論に基づく空間分析を実行します。

主な機能:
- 道路ネットワークの取得と処理
- Space Syntax指標の計算
- ネットワークの可視化
- 分析結果の出力

OSMnx v2.0対応版
"""

import logging

# バージョン情報
__version__ = "0.1.1"
__author__ = "Space Syntax Analyzer Development Team"
__email__ = "dev@spacesyntax.analyzer"

# メインクラスのインポート
from .core.analyzer import SpaceSyntaxAnalyzer, analyze_place_simple
from .core.metrics import SpaceSyntaxMetrics
from .core.network import (
    NetworkManager,
    create_bbox_from_center,
    get_network_from_query,
)
from .core.visualization import NetworkVisualizer

# ユーティリティ関数
from .utils.helpers import format_coordinates, setup_logging, validate_bbox

# 全てのエクスポート
__all__ = [
    # メインクラス
    'SpaceSyntaxAnalyzer',
    'NetworkManager',
    'SpaceSyntaxMetrics',
    'NetworkVisualizer',

    # 便利関数
    'analyze_place_simple',
    'create_bbox_from_center',
    'get_network_from_query',
    'setup_logging',
    'validate_bbox',
    'format_coordinates',

    # バージョン情報
    '__version__'
]

# デフォルトロガーの設定
def _configure_default_logging():
    """デフォルトのログ設定"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# 初期化時にロギング設定
_configure_default_logging()

# パッケージレベルのロガー
logger = logging.getLogger(__name__)
logger.info(f"Space Syntax Analyzer v{__version__} 初期化完了")
