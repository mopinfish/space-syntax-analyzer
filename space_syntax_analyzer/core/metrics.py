"""
メトリクス計算モジュール

スペースシンタックス理論に基づく各種指標の計算を行います。
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class ConnectivityMetrics:
    """回遊性（Connectivity）に関する指標計算クラス"""
    
    @staticmethod
    def calculate_mu_index(graph: nx.Graph) -> int:
        """
        回路指数（μ）を計算: μ = e - ν + p
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            回路指数
        """
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        num_components = nx.number_connected_components(graph)
        
        mu = num_edges - num_nodes + num_components
        return max(0, mu)  # 負の値は0とする
        
    @staticmethod
    def calculate_alpha_index(graph: nx.Graph) -> float:
        """
        α指数を計算: α = μ / (2ν - 5) × 100
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            α指数（パーセント）
        """
        if graph.number_of_nodes() < 3:
            return 0.0
            
        mu = ConnectivityMetrics.calculate_mu_index(graph)
        num_nodes = graph.number_of_nodes()
        
        denominator = 2 * num_nodes - 5
        if denominator <= 0:
            return 0.0
            
        alpha = (mu / denominator) * 100
        return min(100.0, alpha)  # 100%を上限とする
        
    @staticmethod
    def calculate_beta_index(graph: nx.Graph) -> float:
        """
        β指数を計算: β = e / ν
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            β指数
        """
        if graph.number_of_nodes() == 0:
            return 0.0
            
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        
        return num_edges / num_nodes
        
    @staticmethod
    def calculate_gamma_index(graph: nx.Graph) -> float:
        """
        γ指数を計算: γ = e / 3(ν - 2) × 100
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            γ指数（パーセント）
        """
        if graph.number_of_nodes() < 3:
            return 0.0
            
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        
        denominator = 3 * (num_nodes - 2)
        if denominator <= 0:
            return 0.0
            
        gamma = (num_edges / denominator) * 100
        return min(100.0, gamma)  # 100%を上限とする


class AccessibilityMetrics:
    """アクセス性（Accessibility）に関する指標計算クラス"""
    
    @staticmethod
    def calculate_average_shortest_path(graph: nx.Graph) -> float:
        """
        道路網全体の平均最短距離（Di）を計算
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            平均最短距離（メートル）
        """
        if not nx.is_connected(graph):
            # 最大連結成分のみで計算
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc)
            
        try:
            # 全ペア間の最短距離を計算（エッジの重みを使用）
            path_lengths = dict(nx.all_pairs_shortest_path_length(graph, weight='length'))
            
            total_distance = 0.0
            total_pairs = 0
            
            for source, targets in path_lengths.items():
                for target, distance in targets.items():
                    if source != target:
                        total_distance += distance
                        total_pairs += 1
                        
            if total_pairs == 0:
                return 0.0
                
            return total_distance / total_pairs
            
        except Exception as e:
            logger.warning(f"最短距離計算エラー: {e}")
            return 0.0
            
    @staticmethod
    def calculate_road_density(graph: nx.Graph, area_ha: Optional[float] = None) -> float:
        """
        道路密度（Dl）を計算: L / S [m/ha]
        
        Args:
            graph: ネットワークグラフ
            area_ha: 面積（ヘクタール）
            
        Returns:
            道路密度（m/ha）
        """
        if area_ha is None or area_ha <= 0:
            return 0.0
            
        total_length = sum(
            data.get('length', 0) for _, _, data in graph.edges(data=True)
        )
        
        return total_length / area_ha
        
    @staticmethod
    def calculate_intersection_density(graph: nx.Graph, area_ha: Optional[float] = None) -> float:
        """
        交差点密度（Dc）を計算: νc / S [n/ha]
        
        Args:
            graph: ネットワークグラフ
            area_ha: 面積（ヘクタール）
            
        Returns:
            交差点密度（n/ha）
        """
        if area_ha is None or area_ha <= 0:
            return 0.0
            
        # 次数が3以上のノードを交差点とする
        intersections = [node for node, degree in graph.degree() if degree >= 3]
        
        return len(intersections) / area_ha


class CircuityMetrics:
    """迂回性（Circuity）に関する指標計算クラス"""
    
    @staticmethod
    def calculate_average_circuity(graph: nx.Graph) -> float:
        """
        道路網全体の平均迂回率（A）を計算
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            平均迂回率
        """
        if not nx.is_connected(graph):
            # 最大連結成分のみで計算
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc)
            
        try:
            # ノードの座標を取得
            node_coords = {}
            for node, data in graph.nodes(data=True):
                if 'x' in data and 'y' in data:
                    node_coords[node] = (data['x'], data['y'])
                    
            if len(node_coords) < 2:
                return 0.0
                
            # 全ペア間の最短距離（道路距離）を計算
            path_lengths = dict(nx.all_pairs_shortest_path_length(graph, weight='length'))
            
            circuity_ratios = []
            
            for source in node_coords:
                for target in node_coords:
                    if source != target and target in path_lengths[source]:
                        # 道路距離
                        road_distance = path_lengths[source][target]
                        
                        # 直線距離
                        euclidean_distance = euclidean(
                            node_coords[source], 
                            node_coords[target]
                        )
                        
                        if euclidean_distance > 0:
                            circuity = road_distance / euclidean_distance
                            circuity_ratios.append(circuity)
                            
            if not circuity_ratios:
                return 0.0
                
            return np.mean(circuity_ratios)
            
        except Exception as e:
            logger.warning(f"迂回率計算エラー: {e}")
            return 0.0


class SpaceSyntaxMetrics:
    """スペースシンタックス指標の統合計算クラス"""
    
    def __init__(self) -> None:
        self.connectivity = ConnectivityMetrics()
        self.accessibility = AccessibilityMetrics()
        self.circuity = CircuityMetrics()
        
    def calculate_all_metrics(
        self,
        graph: nx.Graph,
        area_ha: float | None = None
    ) -> dict[str, Any]:
        """
        全ての指標を計算
        
        Args:
            graph: ネットワークグラフ
            area_ha: 面積（ヘクタール）
            
        Returns:
            全指標の計算結果
        """
        if graph.number_of_nodes() == 0:
            return self._empty_metrics()
            
        try:
            # 基本統計
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            total_length = sum(
                data.get('length', 0) for _, _, data in graph.edges(data=True)
            )
            
            # 回遊性指標
            mu_index = self.connectivity.calculate_mu_index(graph)
            alpha_index = self.connectivity.calculate_alpha_index(graph)
            beta_index = self.connectivity.calculate_beta_index(graph)
            gamma_index = self.connectivity.calculate_gamma_index(graph)
            
            # アクセス性指標
            avg_shortest_path = self.accessibility.calculate_average_shortest_path(graph)
            road_density = self.accessibility.calculate_road_density(graph, area_ha)
            intersection_density = self.accessibility.calculate_intersection_density(graph, area_ha)
            
            # 迂回性指標
            avg_circuity = self.circuity.calculate_average_circuity(graph)
            
            # 面積あたり回路指数
            mu_per_ha = mu_index / area_ha if area_ha and area_ha > 0 else 0.0
            
            return {
                # 基本統計
                "nodes": num_nodes,
                "edges": num_edges,
                "total_length_m": total_length,
                "area_ha": area_ha,
                
                # 回遊性指標
                "mu_index": mu_index,
                "mu_per_ha": mu_per_ha,
                "alpha_index": alpha_index,
                "beta_index": beta_index,
                "gamma_index": gamma_index,
                
                # アクセス性指標
                "avg_shortest_path": avg_shortest_path,
                "road_density": road_density,
                "intersection_density": intersection_density,
                
                # 迂回性指標
                "avg_circuity": avg_circuity,
            }
            
        except Exception as e:
            logger.error(f"指標計算エラー: {e}")
            return self._empty_metrics()
            
    def _empty_metrics(self) -> dict[str, Any]:
        """空の指標辞書を返す"""
        return {
            "nodes": 0,
            "edges": 0,
            "total_length_m": 0.0,
            "area_ha": 0.0,
            "mu_index": 0,
            "mu_per_ha": 0.0,
            "alpha_index": 0.0,
            "beta_index": 0.0,
            "gamma_index": 0.0,
            "avg_shortest_path": 0.0,
            "road_density": 0.0,
            "intersection_density": 0.0,
            "avg_circuity": 0.0,
        }