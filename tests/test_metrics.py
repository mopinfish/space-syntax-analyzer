"""
メトリクス計算のテストモジュール
"""

import networkx as nx
import numpy as np
import pytest

from space_syntax_analyzer.core.metrics import (
    AccessibilityMetrics,
    CircuityMetrics,
    ConnectivityMetrics,
    SpaceSyntaxMetrics,
)


class TestConnectivityMetrics:
    """ConnectivityMetricsのテストクラス"""

    def test_mu_index_simple_cycle(self):
        """単純な環状グラフの回路指数テスト"""
        graph = nx.cycle_graph(5)
        mu = ConnectivityMetrics.calculate_mu_index(graph)
        # 5ノード、5エッジ、1連結成分 → μ = 5 - 5 + 1 = 1
        assert mu == 1

    def test_mu_index_tree(self):
        """木構造グラフの回路指数テスト"""
        graph = nx.path_graph(5)  # 線形グラフ（木構造）
        mu = ConnectivityMetrics.calculate_mu_index(graph)
        # 5ノード、4エッジ、1連結成分 → μ = 4 - 5 + 1 = 0
        assert mu == 0

    def test_alpha_index_bounds(self):
        """α指数の境界値テスト"""
        # 完全グラフ
        complete_graph = nx.complete_graph(5)
        alpha = ConnectivityMetrics.calculate_alpha_index(complete_graph)
        assert 0 <= alpha <= 100

        # 木構造
        tree_graph = nx.path_graph(5)
        alpha_tree = ConnectivityMetrics.calculate_alpha_index(tree_graph)
        assert alpha_tree == 0

    def test_beta_index_calculation(self):
        """β指数の計算テスト"""
        # スターグラフ（中心ノードの次数が高い）
        star_graph = nx.star_graph(4)
        beta = ConnectivityMetrics.calculate_beta_index(star_graph)
        # 5ノード、4エッジ → β = 4/5 = 0.8
        assert abs(beta - 0.8) < 0.001

    def test_gamma_index_complete_graph(self):
        """γ指数の完全グラフテスト"""
        complete_graph = nx.complete_graph(4)
        gamma = ConnectivityMetrics.calculate_gamma_index(complete_graph)
        # 完全グラフではγ = 100%に近い値になるはず
        assert gamma > 90


class TestAccessibilityMetrics:
    """AccessibilityMetricsのテストクラス"""

    def setup_method(self):
        """テスト用ネットワークの準備"""
        self.simple_graph = nx.Graph()
        self.simple_graph.add_node(1, x=0, y=0)
        self.simple_graph.add_node(2, x=100, y=0)
        self.simple_graph.add_node(3, x=200, y=0)
        self.simple_graph.add_edge(1, 2, length=100)
        self.simple_graph.add_edge(2, 3, length=100)

    def test_road_density_calculation(self):
        """道路密度計算テスト"""
        density = AccessibilityMetrics.calculate_road_density(
            self.simple_graph, area_ha=1.0
        )
        # 総延長200m、面積1ha → 密度 = 200m/ha
        assert density == 200.0

    def test_intersection_density_calculation(self):
        """交差点密度計算テスト"""
        # T字路を作成
        intersection_graph = nx.Graph()
        intersection_graph.add_edge(1, 2)  # ノード2が中心
        intersection_graph.add_edge(2, 3)
        intersection_graph.add_edge(2, 4)  # ノード2は次数3（交差点）

        density = AccessibilityMetrics.calculate_intersection_density(
            intersection_graph, area_ha=1.0
        )
        assert density == 1.0  # 1交差点 / 1ha

    def test_average_shortest_path(self):
        """平均最短距離計算テスト"""
        avg_path = AccessibilityMetrics.calculate_average_shortest_path(
            self.simple_graph
        )
        # 線形グラフの平均最短距離は計算可能
        assert avg_path > 0


class TestCircuityMetrics:
    """CircuityMetricsのテストクラス"""

    def setup_method(self):
        """テスト用ネットワークの準備"""
        # 直線的な道路
        self.straight_graph = nx.Graph()
        self.straight_graph.add_node(1, x=0, y=0)
        self.straight_graph.add_node(2, x=100, y=0)
        self.straight_graph.add_edge(1, 2, length=100)

        # 迂回する道路
        self.detour_graph = nx.Graph()
        self.detour_graph.add_node(1, x=0, y=0)
        self.detour_graph.add_node(2, x=50, y=50)
        self.detour_graph.add_node(3, x=100, y=0)
        self.detour_graph.add_edge(1, 2, length=70.7)  # √(50²+50²)
        self.detour_graph.add_edge(2, 3, length=70.7)

    def test_straight_road_circuity(self):
        """直線道路の迂回率テスト"""
        circuity = CircuityMetrics.calculate_average_circuity(self.straight_graph)
        # 直線道路なので迂回率は1.0に近い
        assert abs(circuity - 1.0) < 0.1

    def test_detour_road_circuity(self):
        """迂回道路の迂回率テスト"""
        circuity = CircuityMetrics.calculate_average_circuity(self.detour_graph)
        # 迂回があるので迂回率は1.0より大きい
        assert circuity > 1.0


class TestSpaceSyntaxMetrics:
    """SpaceSyntaxMetrics統合テストクラス"""

    def setup_method(self):
        """テスト用データの準備"""
        self.metrics_calculator = SpaceSyntaxMetrics()

        # テスト用グラフ（格子状）
        self.grid_graph = nx.grid_2d_graph(3, 3)
        # 座標データを追加
        for node in self.grid_graph.nodes():
            self.grid_graph.nodes[node]['x'] = node[0] * 100
            self.grid_graph.nodes[node]['y'] = node[1] * 100

        # エッジに長さを追加
        for u, v in self.grid_graph.edges():
            self.grid_graph.edges[u, v]['length'] = 100

    def test_calculate_all_metrics(self):
        """全指標計算テスト"""
        results = self.metrics_calculator.calculate_all_metrics(
            self.grid_graph, area_ha=4.0
        )

        # 基本統計の確認
        assert results['nodes'] == 9
        assert results['edges'] == 12
        assert results['area_ha'] == 4.0

        # 指標の存在確認
        required_metrics = [
            'mu_index',
            'alpha_index',
            'beta_index',
            'gamma_index',
            'avg_shortest_path',
            'road_density',
            'intersection_density',
            'avg_circuity',
        ]

        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))

    def test_empty_graph_handling(self):
        """空グラフの処理テスト"""
        empty_graph = nx.Graph()
        results = self.metrics_calculator.calculate_all_metrics(empty_graph)

        # 空のメトリクスが返されることを確認
        assert results['nodes'] == 0
        assert results['edges'] == 0
        assert all(value == 0 or value == 0.0 for value in results.values())

    def test_disconnected_graph_handling(self):
        """非連結グラフの処理テスト"""
        # 2つの分離したコンポーネントを作成
        disconnected_graph = nx.Graph()
        disconnected_graph.add_edge(1, 2, length=100)
        disconnected_graph.add_edge(3, 4, length=100)

        # 座標を追加
        coords = {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (300, 0)}
        for node, (x, y) in coords.items():
            disconnected_graph.nodes[node]['x'] = x
            disconnected_graph.nodes[node]['y'] = y

        # エラーなく処理されることを確認
        results = self.metrics_calculator.calculate_all_metrics(
            disconnected_graph, area_ha=1.0
        )

        assert results['nodes'] == 4
        assert results['edges'] == 2


if __name__ == "__main__":
    pytest.main([__file__])
