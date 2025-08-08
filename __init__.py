"""
Space Syntax Analyzer - スペースシンタックス理論に基づく都市空間分析ライブラリ

このライブラリは、Bill Hillierらによって開発されたスペースシンタックス理論を
基盤とし、都市の道路ネットワークの空間構造を定量的に分析するツールです。
"""

from __future__ import annotations

from .core.analyzer import SpaceSyntaxAnalyzer
from .core.metrics import (
    AccessibilityMetrics,
    CircuityMetrics,
    ConnectivityMetrics,
    SpaceSyntaxMetrics,
)
from .core.network import NetworkManager
from .core.visualization import NetworkVisualizer

__version__ = "0.1.0"
__author__ = "Space Syntax Analyzer Team"
__email__ = "contact@space-syntax-analyzer.org"
__license__ = "MIT"

__all__ = [
    "SpaceSyntaxAnalyzer",
    "NetworkManager", 
    "NetworkVisualizer",
    "SpaceSyntaxMetrics",
    "ConnectivityMetrics",
    "AccessibilityMetrics", 
    "CircuityMetrics",
]

# ライブラリの基本設定
DEFAULT_CRS = "EPSG:4326"
DEFAULT_WIDTH_THRESHOLD = 4.0  # 道路幅員の閾値（メートル）
DEFAULT_NETWORK_TYPE = "drive"  # OSMnxのネットワークタイプ