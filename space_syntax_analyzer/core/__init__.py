"""
コアモジュール

Space Syntax分析の中核機能を提供します。
"""

from .analyzer import SpaceSyntaxAnalyzer, analyze_place_simple
from .network import NetworkManager, create_bbox_from_center, get_network_from_query

# metrics.py と visualization.py が存在する場合のみインポート
try:
    from .metrics import SpaceSyntaxMetrics
except ImportError:
    SpaceSyntaxMetrics = None

try:
    from .visualization import NetworkVisualizer
except ImportError:
    NetworkVisualizer = None

__all__ = [
    'SpaceSyntaxAnalyzer',
    'analyze_place_simple',
    'NetworkManager',
    'create_bbox_from_center',
    'get_network_from_query',
    'SpaceSyntaxMetrics',
    'NetworkVisualizer'
]
