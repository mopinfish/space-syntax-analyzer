"""
ユーティリティ関数モジュール

共通的に使用される補助関数を提供します。
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def validate_graph(graph: nx.Graph) -> bool:
    """
    グラフの妥当性を検証
    
    Args:
        graph: 検証対象のグラフ
        
    Returns:
        グラフが有効かどうか
    """
    if not isinstance(graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        return False
        
    if graph.number_of_nodes() == 0:
        return False
        
    # 座標データの存在確認
    has_coordinates = any(
        'x' in data and 'y' in data 
        for _, data in graph.nodes(data=True)
    )
    
    if not has_coordinates:
        logger.warning("ノードに座標データがありません")
        
    return True


def calculate_graph_bounds(
    graph: nx.Graph,
) -> tuple[float, float, float, float] | None:
    """
    グラフの境界を計算
    
    Args:
        graph: ネットワークグラフ
        
    Returns:
        境界座標 (min_x, min_y, max_x, max_y) または None
    """
    try:
        coordinates = []
        for _, data in graph.nodes(data=True):
            if 'x' in data and 'y' in data:
                coordinates.append((data['x'], data['y']))
                
        if not coordinates:
            return None
            
        coords_array = np.array(coordinates)
        min_x, min_y = coords_array.min(axis=0)
        max_x, max_y = coords_array.max(axis=0)
        
        return (min_x, min_y, max_x, max_y)
        
    except Exception as e:
        logger.error(f"境界計算エラー: {e}")
        return None


def normalize_metrics(
    metrics: Dict[str, Any],
    reference_values: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    指標値を正規化
    
    Args:
        metrics: 正規化対象の指標
        reference_values: 参照値（最大値として使用）
        
    Returns:
        正規化された指標
    """
    normalized = metrics.copy()
    
    # デフォルトの参照値
    default_references = {
        'alpha_index': 100.0,
        'gamma_index': 100.0,
        'beta_index': 5.0,
        'avg_circuity': 3.0,
    }
    
    if reference_values:
        default_references.update(reference_values)
        
    for key, value in metrics.items():
        if key in default_references and isinstance(value, (int, float)):
            ref_value = default_references[key]
            normalized[f"{key}_normalized"] = min(value / ref_value, 1.0)
            
    return normalized


def classify_network_type(metrics: dict[str, Any]) -> str:
    """
    メトリクスに基づいてネットワークタイプを分類
    
    Args:
        metrics: 計算された指標
        
    Returns:
        ネットワークタイプ（'格子型', '放射型', '樹状型', '不定型'）
    """
    alpha = metrics.get('alpha_index', 0)
    beta = metrics.get('beta_index', 0)
    gamma = metrics.get('gamma_index', 0)
    
    # 分類ルール（研究資料に基づく）
    if alpha > 30 and gamma > 60:
        return "格子型"
    elif beta > 1.5 and alpha > 20:
        return "放射型"
    elif alpha < 10 and beta < 1.2:
        return "樹状型"
    else:
        return "不定型"


def generate_comparison_summary(
    major_results: Dict[str, Any],
    full_results: Dict[str, Any]
) -> Dict[str, str]:
    """
    主要道路と全道路の比較サマリーを生成
    
    Args:
        major_results: 主要道路の分析結果
        full_results: 全道路の分析結果
        
    Returns:
        比較サマリー
    """
    summary = {}
    
    # 回遊性の変化
    alpha_change = full_results.get('alpha_index', 0) - major_results.get('alpha_index', 0)
    if alpha_change > 5:
        summary['connectivity'] = "細街路により回遊性が大幅に向上"
    elif alpha_change > 0:
        summary['connectivity'] = "細街路により回遊性がやや向上"
    elif alpha_change < -5:
        summary['connectivity'] = "細街路により回遊性が低下"
    else:
        summary['connectivity'] = "細街路による回遊性への影響は軽微"
        
    # アクセス性の変化
    distance_change = full_results.get('avg_shortest_path', 0) - major_results.get('avg_shortest_path', 0)
    if distance_change < -50:
        summary['accessibility'] = "細街路によりアクセス性が大幅に向上"
    elif distance_change < 0:
        summary['accessibility'] = "細街路によりアクセス性がやや向上"
    elif distance_change > 50:
        summary['accessibility'] = "細街路によりアクセス性が低下"
    else:
        summary['accessibility'] = "細街路によるアクセス性への影響は軽微"
        
    # 迂回性の変化
    circuity_change = full_results.get('avg_circuity', 0) - major_results.get('avg_circuity', 0)
    if circuity_change < -0.2:
        summary['circuity'] = "細街路により迂回性が改善"
    elif circuity_change > 0.2:
        summary['circuity'] = "細街路により迂回性が悪化"
    else:
        summary['circuity'] = "細街路による迂回性への影響は軽微"
        
    # ネットワークタイプ
    major_type = classify_network_type(major_results)
    full_type = classify_network_type(full_results)
    summary['network_type'] = f"主要道路: {major_type}, 全道路: {full_type}"
    
    return summary


def create_bbox_from_center(
    center_lat: float,
    center_lon: float,
    distance_km: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    中心点から指定距離の境界ボックスを作成
    
    Args:
        center_lat: 中心緯度
        center_lon: 中心経度
        distance_km: 距離（キロメートル）
        
    Returns:
        境界ボックス (north, south, east, west)
    """
    # 緯度1度 ≈ 111km, 経度1度 ≈ 111km * cos(緯度)
    lat_degree = distance_km / 111.0
    lon_degree = distance_km / (111.0 * np.cos(np.radians(center_lat)))
    
    north = center_lat + lat_degree
    south = center_lat - lat_degree
    east = center_lon + lon_degree
    west = center_lon - lon_degree
    
    return (north, south, east, west)


def setup_logging(level: str = "INFO") -> None:
    """
    ロギングの設定
    
    Args:
        level: ログレベル
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
