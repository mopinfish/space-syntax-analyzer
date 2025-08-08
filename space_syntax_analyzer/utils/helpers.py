"""
ユーティリティ関数とヘルパー (OSMnx v2.0対応)

共通的に使用される便利関数を提供します。
"""

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO", format_string: str | None = None) -> logging.Logger:
    """
    ロギング設定

    Args:
        level: ログレベル ("DEBUG", "INFO", "WARNING", "ERROR")
        format_string: カスタムフォーマット文字列

    Returns:
        設定されたロガー
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 有効なログレベルのチェック
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        level = 'INFO'  # デフォルトにフォールバック

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # OSMnxのログレベルも調整
    ox_logger = logging.getLogger('osmnx')
    ox_logger.setLevel(logging.WARNING)  # OSMnxのログを抑制

    logger = logging.getLogger('space_syntax_analyzer')
    logger.info(f"ロギング設定完了: レベル={level}")
    return logger


def validate_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """
    bboxの妥当性を検証

    Args:
        bbox: (left, bottom, right, top) 形式のbbox

    Returns:
        有効な場合True
    """
    try:
        left, bottom, right, top = bbox

        # 基本的な範囲チェック
        if not (-180 <= left <= 180 and -180 <= right <= 180):
            return False
        if not (-90 <= bottom <= 90 and -90 <= top <= 90):
            return False

        # 大小関係チェック
        if left >= right or bottom >= top:
            return False

        # 範囲が大きすぎないかチェック（度数で5度以内）
        if (right - left) > 5 or (top - bottom) > 5:
            logging.warning(f"bboxが大きすぎます: {bbox}")
            return False

        return True

    except (TypeError, ValueError):
        return False


def format_coordinates(lat: float, lon: float, precision: int = 4) -> str:
    """
    座標を読みやすい形式でフォーマット

    Args:
        lat: 緯度
        lon: 経度
        precision: 小数点以下の桁数

    Returns:
        フォーマットされた座標文字列
    """
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    return f"{abs(lat):.{precision}f}°{lat_dir}, {abs(lon):.{precision}f}°{lon_dir}"


def calculate_bbox_area(bbox: tuple[float, float, float, float]) -> float:
    """
    bboxの面積を計算（平方キロメートル）

    Args:
        bbox: (left, bottom, right, top) 形式のbbox

    Returns:
        面積（km²）
    """
    try:
        left, bottom, right, top = bbox

        # 中心緯度での補正
        center_lat = (bottom + top) / 2
        lat_correction = np.cos(np.radians(center_lat))

        # 緯度経度差をkmに変換
        lat_km = (top - bottom) * 111.0  # 1度 ≈ 111km
        lon_km = (right - left) * 111.0 * lat_correction

        return lat_km * lon_km

    except (TypeError, ValueError, AttributeError) as e:
        logging.error(f"bbox面積計算エラー: {e}")
        return 0.0


def create_analysis_summary(results: dict[str, Any]) -> dict[str, Any]:
    """
    分析結果の要約を作成

    Args:
        results: 分析結果

    Returns:
        要約辞書
    """
    try:
        summary = {}

        # メタデータ
        metadata = results.get('metadata', {})
        summary['query'] = metadata.get('query', 'N/A')
        summary['network_type'] = metadata.get('network_type', 'N/A')
        summary['analysis_status'] = metadata.get('analysis_status', 'N/A')

        # 主要道路データ
        major = results.get('major_network')
        if major:
            summary['major_nodes'] = major.get('node_count', 0)
            summary['major_edges'] = major.get('edge_count', 0)
            summary['major_alpha_index'] = major.get('alpha_index', 0)
            summary['major_connectivity'] = major.get('connectivity_ratio', 0)
        else:
            summary['major_nodes'] = 0
            summary['major_edges'] = 0
            summary['major_alpha_index'] = 0
            summary['major_connectivity'] = 0

        # 全道路データ
        full = results.get('full_network')
        if full:
            summary['full_nodes'] = full.get('node_count', 0)
            summary['full_edges'] = full.get('edge_count', 0)
            summary['full_alpha_index'] = full.get('alpha_index', 0)
            summary['full_connectivity'] = full.get('connectivity_ratio', 0)
        else:
            summary['full_nodes'] = 0
            summary['full_edges'] = 0
            summary['full_alpha_index'] = 0
            summary['full_connectivity'] = 0

        # 比較指標
        if summary['full_nodes'] > 0:
            summary['major_ratio'] = summary['major_nodes'] / summary['full_nodes'] * 100
        else:
            summary['major_ratio'] = 0

        return summary

    except Exception as e:
        logging.error(f"要約作成エラー: {e}")
        return {'error': str(e)}


def export_summary_table(results_list: list[dict[str, Any]], filepath: str) -> bool:
    """
    複数の分析結果をテーブル形式で出力

    Args:
        results_list: 分析結果のリスト
        filepath: 出力先パス

    Returns:
        出力成功時True
    """
    try:
        summaries = [create_analysis_summary(result) for result in results_list]
        df = pd.DataFrame(summaries)

        if filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logging.info(f"要約テーブル出力完了: {filepath}")
        return True

    except Exception as e:
        logging.error(f"要約テーブル出力エラー: {e}")
        return False


def check_osmnx_version() -> dict[str, str]:
    """
    OSMnxのバージョン情報を確認

    Returns:
        バージョン情報辞書
    """
    try:
        import geopandas as gpd
        import networkx as nx
        import osmnx as ox

        return {
            'osmnx': ox.__version__,
            'networkx': nx.__version__,
            'geopandas': gpd.__version__,
            'compatibility': 'v2.0対応' if ox.__version__.startswith('2.') else '要確認'
        }

    except Exception as e:
        return {'error': str(e)}


def estimate_processing_time(bbox: tuple[float, float, float, float]) -> str:
    """
    処理時間の大まかな見積もり

    Args:
        bbox: 対象のbbox

    Returns:
        処理時間見積もり文字列
    """
    try:
        area = calculate_bbox_area(bbox)

        if area < 1:
            return "< 30秒"
        elif area < 5:
            return "30秒 - 2分"
        elif area < 25:
            return "2分 - 10分"
        else:
            return "10分以上（大きなエリア）"

    except (TypeError, ValueError, AttributeError) as e:
        logging.error(f"処理時間見積もりエラー: {e}")
        return "不明"


def create_network_comparison_report(major_net: nx.MultiDiGraph | None,
                                   full_net: nx.MultiDiGraph | None) -> str:
    """
    ネットワーク比較レポートを作成

    Args:
        major_net: 主要道路ネットワーク
        full_net: 全道路ネットワーク

    Returns:
        比較レポート文字列
    """
    try:
        lines = ["ネットワーク比較レポート", "=" * 30, ""]

        if major_net is None and full_net is None:
            lines.append("❌ 両方のネットワークが取得できませんでした")
            return "\n".join(lines)

        if full_net:
            full_stats = f"全道路: {len(full_net.nodes)} ノード, {len(full_net.edges)} エッジ"
            lines.append(full_stats)

        if major_net:
            major_stats = f"主要道路: {len(major_net.nodes)} ノード, {len(major_net.edges)} エッジ"
            lines.append(major_stats)

            # 比率計算
            if full_net:
                node_ratio = len(major_net.nodes) / len(full_net.nodes) * 100
                edge_ratio = len(major_net.edges) / len(full_net.edges) * 100
                lines.extend([
                    "",
                    "主要道路比率:",
                    f"  ノード: {node_ratio:.1f}%",
                    f"  エッジ: {edge_ratio:.1f}%"
                ])

        return "\n".join(lines)

    except Exception as e:
        logging.error(f"レポート作成エラー: {e}")
        return f"レポート作成エラー: {e}"


def generate_comparison_summary(major_network_results: dict[str, Any],
                              full_network_results: dict[str, Any]) -> dict[str, Any]:
    """
    ネットワーク比較サマリーを生成（後方互換性のため）

    Args:
        major_network_results: 主要道路ネットワークの分析結果
        full_network_results: 全道路ネットワークの分析結果

    Returns:
        比較サマリーの辞書
    """
    try:
        summary = {}

        # ノード数の比較
        major_nodes = major_network_results.get('node_count', 0)
        full_nodes = full_network_results.get('node_count', 0)

        summary['主要道路ノード数'] = major_nodes
        summary['全道路ノード数'] = full_nodes
        summary['主要道路比率'] = f"{(major_nodes / full_nodes * 100):.1f}%" if full_nodes > 0 else "0%"

        # エッジ数の比較
        major_edges = major_network_results.get('edge_count', 0)
        full_edges = full_network_results.get('edge_count', 0)

        summary['主要道路エッジ数'] = major_edges
        summary['全道路エッジ数'] = full_edges
        summary['主要エッジ比率'] = f"{(major_edges / full_edges * 100):.1f}%" if full_edges > 0 else "0%"

        # 平均次数の比較
        major_degree = major_network_results.get('avg_degree', 0)
        full_degree = full_network_results.get('avg_degree', 0)

        summary['主要道路平均次数'] = f"{major_degree:.2f}"
        summary['全道路平均次数'] = f"{full_degree:.2f}"

        return summary

    except Exception as e:
        return {'エラー': str(e)}


# デバッグ用のヘルパー関数
def debug_network_info(G: nx.MultiDiGraph, name: str = "Network") -> None:
    """
    ネットワークのデバッグ情報を出力

    Args:
        G: 対象ネットワーク
        name: ネットワーク名
    """
    if G is None:
        print(f"{name}: None")
        return

    print(f"\n{name} デバッグ情報:")
    print(f"  ノード数: {len(G.nodes)}")
    print(f"  エッジ数: {len(G.edges)}")
    print(f"  単純化済み: {G.graph.get('simplified', False)}")
    print(f"  CRS: {G.graph.get('crs', 'N/A')}")

    if len(G.nodes) > 0:
        degrees = dict(G.degree())
        print(f"  平均次数: {np.mean(list(degrees.values())):.2f}")
        print(f"  最大次数: {max(degrees.values())}")

    # エッジ属性の例
    if len(G.edges) > 0:
        edge_data = list(G.edges(data=True))[0][2]
        print(f"  エッジ属性例: {list(edge_data.keys())[:5]}")
