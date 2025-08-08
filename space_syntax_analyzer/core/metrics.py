"""
Space Syntax指標計算モジュール

ネットワークからSpace Syntax理論に基づく各種指標を計算します。
"""

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SpaceSyntaxMetrics:
    """Space Syntax指標計算クラス"""

    def __init__(self):
        """メトリクス計算クラスを初期化"""
        self.logger = logging.getLogger(__name__)

    def calculate_all_metrics(self, G: nx.MultiDiGraph) -> dict[str, Any]:
        """
        全てのSpace Syntax指標を計算

        Args:
            G: 分析対象のネットワーク

        Returns:
            計算された指標の辞書
        """
        try:
            metrics = {}

            # 基本的な指標
            metrics.update(self._calculate_basic_metrics(G))

            # Space Syntax指標
            metrics.update(self._calculate_space_syntax_indices(G))

            # 道路密度と迂回率
            metrics.update(self._calculate_density_metrics(G))

            # 連結性指標
            metrics.update(self._calculate_connectivity_metrics(G))

            self.logger.info(f"Space Syntax指標計算完了: {len(metrics)} 指標")
            return metrics

        except Exception as e:
            self.logger.error(f"Space Syntax指標計算エラー: {e}")
            return self._get_default_metrics()

    def _calculate_basic_metrics(self, G: nx.MultiDiGraph) -> dict[str, float]:
        """基本的なネットワーク指標を計算"""
        try:
            n_nodes = len(G.nodes)
            n_edges = len(G.edges)

            if n_nodes == 0:
                return {'node_count': 0, 'edge_count': 0, 'avg_degree': 0.0, 'density': 0.0}

            # 次数統計
            degrees = dict(G.degree())
            avg_degree = np.mean(list(degrees.values()))

            # ネットワーク密度
            density = nx.density(G)

            return {
                'node_count': n_nodes,
                'edge_count': n_edges,
                'avg_degree': float(avg_degree),
                'max_degree': max(degrees.values()) if degrees else 0,
                'min_degree': min(degrees.values()) if degrees else 0,
                'density': float(density)
            }

        except Exception as e:
            self.logger.error(f"基本指標計算エラー: {e}")
            return {'node_count': 0, 'edge_count': 0, 'avg_degree': 0.0, 'density': 0.0}

    def _calculate_space_syntax_indices(self, G: nx.MultiDiGraph) -> dict[str, float]:
        """Space Syntax指標（α、β、γ指数）を計算"""
        try:
            n_nodes = len(G.nodes)
            n_edges = len(G.edges)

            if n_nodes < 3:
                return {
                    'alpha_index': 0.0,
                    'beta_index': 0.0,
                    'gamma_index': 0.0
                }

            # α指数: 実際の閉路数 / 最大可能閉路数
            max_circuits = 2 * n_nodes - 5
            actual_circuits = max(0, n_edges - n_nodes + 1)  # オイラーの公式より
            alpha_index = (actual_circuits / max_circuits * 100) if max_circuits > 0 else 0.0

            # β指数: エッジ数 / ノード数
            beta_index = n_edges / n_nodes

            # γ指数: 実際のエッジ数 / 最大可能エッジ数
            max_edges = 3 * (n_nodes - 2)
            gamma_index = n_edges / max_edges if max_edges > 0 else 0.0

            return {
                'alpha_index': float(alpha_index),
                'beta_index': float(beta_index),
                'gamma_index': float(gamma_index)
            }

        except Exception as e:
            self.logger.error(f"Space Syntax指数計算エラー: {e}")
            return {'alpha_index': 0.0, 'beta_index': 0.0, 'gamma_index': 0.0}

    def _calculate_density_metrics(self, G: nx.MultiDiGraph) -> dict[str, float]:
        """密度と迂回率関連の指標を計算"""
        try:
            # 道路の総延長を計算（エッジの長さの合計）
            total_length = 0
            edge_lengths = []

            for _u, _v, data in G.edges(data=True):
                length = data.get('length', 0)
                if length > 0:
                    total_length += length
                    edge_lengths.append(length)

            # 推定エリア面積（ノードの外接矩形から）
            if len(G.nodes) > 0:
                lats = [G.nodes[node].get('y', 0) for node in G.nodes]
                lons = [G.nodes[node].get('x', 0) for node in G.nodes]

                if lats and lons:
                    lat_range = max(lats) - min(lats)
                    lon_range = max(lons) - min(lons)

                    # 緯度経度から大まかな面積を計算（km²）
                    center_lat = (max(lats) + min(lats)) / 2
                    lat_km = lat_range * 111.0  # 1度 ≈ 111km
                    lon_km = lon_range * 111.0 * np.cos(np.radians(center_lat))
                    area_km2 = lat_km * lon_km

                    # 道路密度 (km/km²)
                    road_density = (total_length / 1000) / area_km2 if area_km2 > 0 else 0
                else:
                    road_density = 0
            else:
                road_density = 0

            # 平均迂回率（簡易版）
            avg_circuity = self._calculate_circuity(G) if len(G.edges) > 0 else 1.0

            return {
                'road_density': float(road_density),
                'avg_circuity': float(avg_circuity),
                'total_length': float(total_length),
                'avg_edge_length': float(np.mean(edge_lengths)) if edge_lengths else 0
            }

        except Exception as e:
            self.logger.error(f"密度指標計算エラー: {e}")
            return {'road_density': 0.0, 'avg_circuity': 1.0, 'total_length': 0.0, 'avg_edge_length': 0.0}

    def _calculate_circuity(self, G: nx.MultiDiGraph) -> float:
        """迂回率を計算（簡易版）"""
        try:
            # サンプルノードペアで迂回率を計算
            nodes = list(G.nodes)
            if len(nodes) < 10:
                return 1.0

            # ランダムに10組のノードペアをサンプル
            np.random.seed(42)  # 再現性のため
            sample_pairs = np.random.choice(nodes, size=(min(10, len(nodes)//2), 2), replace=False)

            circuities = []
            for pair in sample_pairs:
                try:
                    if pair[0] != pair[1] and nx.has_path(G, pair[0], pair[1]):
                        # ネットワーク距離
                        path_length = nx.shortest_path_length(G, pair[0], pair[1], weight='length')

                        # 直線距離
                        node1_data = G.nodes[pair[0]]
                        node2_data = G.nodes[pair[1]]
                        if all(key in node1_data for key in ['x', 'y']) and all(key in node2_data for key in ['x', 'y']):
                            euclidean_dist = np.sqrt(
                                (node1_data['x'] - node2_data['x'])**2 +
                                (node1_data['y'] - node2_data['y'])**2
                            ) * 111000  # 度をメートルに概算変換

                            if euclidean_dist > 0:
                                circuity = path_length / euclidean_dist
                                circuities.append(circuity)
                except (nx.NetworkXError, KeyError, ValueError):
                    continue

            return float(np.mean(circuities)) if circuities else 1.0

        except Exception as e:
            self.logger.error(f"迂回率計算エラー: {e}")
            return 1.0

    def _calculate_connectivity_metrics(self, G: nx.MultiDiGraph) -> dict[str, float]:
        """連結性関連の指標を計算"""
        try:
            if len(G.nodes) == 0:
                return {
                    'is_connected': False,
                    'largest_component_size': 0,
                    'connectivity_ratio': 0.0,
                    'num_components': 0
                }

            # 弱連結成分の分析
            weak_components = list(nx.weakly_connected_components(G))
            largest_component_size = len(max(weak_components, key=len))

            # 連結性指標
            is_connected = len(weak_components) == 1
            connectivity_ratio = largest_component_size / len(G.nodes)

            return {
                'is_connected': is_connected,
                'largest_component_size': largest_component_size,
                'connectivity_ratio': float(connectivity_ratio),
                'num_components': len(weak_components)
            }

        except Exception as e:
            self.logger.error(f"連結性指標計算エラー: {e}")
            return {
                'is_connected': False,
                'largest_component_size': 0,
                'connectivity_ratio': 0.0,
                'num_components': 0
            }

    def _get_default_metrics(self) -> dict[str, float]:
        """デフォルトの指標値を返す"""
        return {
            'node_count': 0,
            'edge_count': 0,
            'avg_degree': 0.0,
            'max_degree': 0,
            'min_degree': 0,
            'density': 0.0,
            'alpha_index': 0.0,
            'beta_index': 0.0,
            'gamma_index': 0.0,
            'road_density': 0.0,
            'avg_circuity': 1.0,
            'total_length': 0.0,
            'avg_edge_length': 0.0,
            'is_connected': False,
            'largest_component_size': 0,
            'connectivity_ratio': 0.0,
            'num_components': 0
        }

    def calculate_integration_values(self, G: nx.MultiDiGraph) -> dict[int, float]:
        """
        各ノードのIntegration値を計算（簡易版）

        Args:
            G: 分析対象のネットワーク

        Returns:
            ノードIDをキー、Integration値を値とする辞書
        """
        try:
            integration_values = {}

            if len(G.nodes) < 2:
                return integration_values

            # 各ノードからの最短パス長の平均を計算
            for node in G.nodes:
                try:
                    # 単一始点最短パス
                    path_lengths = nx.single_source_shortest_path_length(G, node)

                    if len(path_lengths) > 1:
                        # 自分自身を除いた平均パス長
                        lengths = [length for target, length in path_lengths.items() if target != node]
                        mean_depth = np.mean(lengths) if lengths else 0

                        # Integration値（平均深度の逆数的な値）
                        integration = 1.0 / (1.0 + mean_depth) if mean_depth > 0 else 0
                        integration_values[node] = integration
                    else:
                        integration_values[node] = 0.0

                except (nx.NetworkXError, ValueError, KeyError):
                    integration_values[node] = 0.0

            self.logger.info(f"Integration値計算完了: {len(integration_values)} ノード")
            return integration_values

        except Exception as e:
            self.logger.error(f"Integration値計算エラー: {e}")
            return {}

    def calculate_choice_values(self, G: nx.MultiDiGraph) -> dict[int, float]:
        """
        各ノードのChoice値を計算（簡易版）

        Args:
            G: 分析対象のネットワーク

        Returns:
            ノードIDをキー、Choice値を値とする辞書
        """
        try:
            choice_values = dict.fromkeys(G.nodes, 0.0)

            if len(G.nodes) < 3:
                return choice_values

            # 中心性（媒介中心性）を使用してChoice値を近似
            betweenness = nx.betweenness_centrality(G, normalized=True)

            for node in G.nodes:
                choice_values[node] = betweenness.get(node, 0.0)

            self.logger.info(f"Choice値計算完了: {len(choice_values)} ノード")
            return choice_values

        except Exception as e:
            self.logger.error(f"Choice値計算エラー: {e}")
            return dict.fromkeys(G.nodes, 0.0)


# 便利関数
def calculate_basic_network_stats(G: nx.MultiDiGraph) -> dict[str, Any]:
    """
    ネットワークの基本統計を計算する便利関数

    Args:
        G: 分析対象のネットワーク

    Returns:
        基本統計の辞書
    """
    metrics = SpaceSyntaxMetrics()
    return metrics._calculate_basic_metrics(G)


def calculate_space_syntax_indices(G: nx.MultiDiGraph) -> dict[str, float]:
    """
    Space Syntax指数を計算する便利関数

    Args:
        G: 分析対象のネットワーク

    Returns:
        Space Syntax指数の辞書
    """
    metrics = SpaceSyntaxMetrics()
    return metrics._calculate_space_syntax_indices(G)
