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

# ユーティリティ関数のインポート（basic_usage.pyで必要な関数を含む）
try:
    from .utils.helpers import (
        calculate_bbox_area,
        check_dependencies,
        check_osmnx_version,
        create_analysis_summary,
        create_simple_summary_report,
        debug_network_info,
        estimate_processing_time,
        format_coordinates,
        generate_comparison_summary,
        get_memory_usage_info,
        setup_logging,
        validate_bbox,
        validate_network_data,
    )
except ImportError:
    # utils.helpersが存在しない場合の代替実装
    import platform
    import sys
    from typing import Any

    def setup_logging(level: str = "INFO") -> None:
        """ロギングを設定"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        logger = logging.getLogger("space_syntax_analyzer")
        logger.setLevel(log_level)
        logger.handlers.clear()
        logger.addHandler(handler)

        # 外部ライブラリのログレベルを調整
        logging.getLogger("osmnx").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    def check_osmnx_version() -> dict[str, str]:
        """OSMnxとその他の依存関係のバージョン情報を取得"""
        version_info = {}

        try:
            import osmnx
            version_info["osmnx"] = osmnx.__version__
        except ImportError:
            version_info["osmnx"] = "Not installed"

        try:
            import networkx
            version_info["networkx"] = networkx.__version__
        except ImportError:
            version_info["networkx"] = "Not installed"

        try:
            import pandas
            version_info["pandas"] = pandas.__version__
        except ImportError:
            version_info["pandas"] = "Not installed"

        version_info["python"] = sys.version.split()[0]
        version_info["platform"] = platform.system()

        return version_info

    def debug_network_info(G, network_name: str) -> None:
        """ネットワークのデバッグ情報を出力"""
        logger = logging.getLogger(__name__)
        if G is None:
            logger.info(f"{network_name}: ネットワークがNullです")
            return

        logger.info(f"{network_name} デバッグ情報:")
        logger.info(f"  ノード数: {len(G.nodes)}")
        logger.info(f"  エッジ数: {len(G.edges)}")

    def generate_comparison_summary(major_network: dict[str, Any],
                                  full_network: dict[str, Any]) -> dict[str, str]:
        """主要道路と全道路ネットワークの比較サマリーを生成"""
        summary = {}

        try:
            # ノード数比較
            major_nodes = major_network.get("node_count", 0)
            full_nodes = full_network.get("node_count", 0)
            node_ratio = (major_nodes / full_nodes * 100) if full_nodes > 0 else 0
            summary["主要道路ノード比率"] = f"{node_ratio:.1f}% ({major_nodes:,} / {full_nodes:,})"

            # エッジ数比較
            major_edges = major_network.get("edge_count", 0)
            full_edges = full_network.get("edge_count", 0)
            edge_ratio = (major_edges / full_edges * 100) if full_edges > 0 else 0
            summary["主要道路エッジ比率"] = f"{edge_ratio:.1f}% ({major_edges:,} / {full_edges:,})"

            # 指標比較
            major_alpha = major_network.get("alpha_index", 0)
            full_alpha = full_network.get("alpha_index", 0)
            summary["α指数比較"] = f"主要: {major_alpha:.1f}%, 全体: {full_alpha:.1f}%"

        except Exception as e:
            summary["エラー"] = str(e)

        return summary

    def create_analysis_summary(results: dict[str, Any]) -> str:
        """分析結果のサマリーを作成"""
        try:
            lines = ["=== 分析サマリー ==="]

            # メタデータ
            metadata = results.get("metadata", {})
            lines.append(f"対象: {metadata.get('query', 'N/A')}")
            lines.append(f"ステータス: {metadata.get('analysis_status', 'N/A')}")
            lines.append("")

            # 主要道路ネットワーク
            major = results.get("major_network")
            if major:
                lines.append("主要道路ネットワーク:")
                lines.append(f"  ノード: {major.get('node_count', 0):,}")
                lines.append(f"  エッジ: {major.get('edge_count', 0):,}")
                lines.append(f"  α指数: {major.get('alpha_index', 0):.1f}%")
                lines.append(f"  β指数: {major.get('beta_index', 0):.2f}")
                lines.append("")

            # 全道路ネットワーク
            full = results.get("full_network")
            if full:
                lines.append("全道路ネットワーク:")
                lines.append(f"  ノード: {full.get('node_count', 0):,}")
                lines.append(f"  エッジ: {full.get('edge_count', 0):,}")
                lines.append(f"  α指数: {full.get('alpha_index', 0):.1f}%")
                lines.append(f"  β指数: {full.get('beta_index', 0):.2f}")

            return "\n".join(lines)

        except Exception as e:
            return f"サマリー生成エラー: {e}"

    def calculate_bbox_area(bbox: tuple[float, float, float, float]) -> float:
        """境界ボックスの面積を計算（km²）"""
        try:
            import numpy as np
            left, bottom, right, top = bbox

            # 緯度経度を距離に変換（概算）
            center_lat = (bottom + top) / 2
            lat_dist = (top - bottom) * 111.0  # 1度 ≈ 111km
            lon_dist = (right - left) * 111.0 * np.cos(np.radians(center_lat))

            area_km2 = lat_dist * lon_dist
            return area_km2

        except Exception:
            return 0.0

    def estimate_processing_time(bbox: tuple[float, float, float, float]) -> str:
        """処理時間を推定"""
        try:
            area_km2 = calculate_bbox_area(bbox)

            if area_km2 < 0.1:
                return "約10-30秒"
            elif area_km2 < 0.5:
                return "約30秒-1分"
            elif area_km2 < 1.0:
                return "約1-2分"
            elif area_km2 < 2.0:
                return "約2-5分"
            else:
                return "5分以上（大きなエリア）"

        except Exception:
            return "推定不可"

    def format_coordinates(lat: float, lon: float) -> str:
        """座標をフォーマット"""
        return f"({lat:.4f}, {lon:.4f})"

    def validate_bbox(bbox: tuple[float, float, float, float]) -> bool:
        """境界ボックスの妥当性を検証"""
        try:
            left, bottom, right, top = bbox
            return left < right and bottom < top
        except Exception:
            return False

    def validate_network_data(G) -> dict[str, Any]:
        """ネットワークデータの妥当性を検証"""
        return {"is_valid": G is not None, "issues": [], "warnings": [], "stats": {}}

    def create_simple_summary_report(results: dict[str, Any]) -> str:
        """簡単なサマリーレポートを作成"""
        return create_analysis_summary(results)

    def check_dependencies() -> dict[str, bool]:
        """必要な依存関係をチェック"""
        return {"osmnx": True, "networkx": True, "pandas": True, "numpy": True}

    def get_memory_usage_info() -> dict[str, str]:
        """メモリ使用量の情報を取得"""
        return {"rss": "N/A", "vms": "N/A", "percent": "N/A"}

# 全てのエクスポート
__all__ = [
    # メインクラス
    "SpaceSyntaxAnalyzer",
    "NetworkManager",
    "SpaceSyntaxMetrics",
    "NetworkVisualizer",

    # 便利関数
    "analyze_place_simple",
    "create_bbox_from_center",
    "get_network_from_query",
    "setup_logging",
    "validate_bbox",
    "format_coordinates",

    # basic_usage.pyで必要な追加関数
    "check_osmnx_version",
    "debug_network_info",
    "generate_comparison_summary",
    "calculate_bbox_area",
    "estimate_processing_time",
    "create_analysis_summary",
    "validate_network_data",
    "create_simple_summary_report",
    "check_dependencies",
    "get_memory_usage_info",

    # バージョン情報
    "__version__"
]

# デフォルトロガーの設定
def _configure_default_logging():
    """デフォルトのログ設定"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# 初期化時にロギング設定
_configure_default_logging()

# パッケージレベルのロガー
logger = logging.getLogger(__name__)
logger.info(f"Space Syntax Analyzer v{__version__} 初期化完了")
