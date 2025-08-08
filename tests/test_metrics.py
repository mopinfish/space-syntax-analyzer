"""
メトリクス計算のテストモジュール（修正版）
"""

import networkx as nx
import pytest

from space_syntax_analyzer.core.metrics import (
    SpaceSyntaxMetrics,
    calculate_basic_network_stats,
    calculate_space_syntax_indices,
)


class TestSpaceSyntaxMetrics:
    """SpaceSyntaxMetricsのテストクラス"""

    def setup_method(self):
        """テスト用データの準備"""
        self.metrics_calculator = SpaceSyntaxMetrics()

        # テスト用グラフ（格子状）
        self.grid_graph = nx.grid_2d_graph(3, 3)
        # 座標データを追加
        for node in self.grid_graph.nodes():
            self.grid_graph.nodes[node]["x"] = node[0] * 100
            self.grid_graph.nodes[node]["y"] = node[1] * 100

        # エッジに長さを追加
        for u, v in self.grid_graph.edges():
            self.grid_graph.edges[u, v]["length"] = 100

    def test_calculate_all_metrics(self):
        """全指標計算テスト"""
        results = self.metrics_calculator.calculate_all_metrics(self.grid_graph)

        # 基本統計の確認
        assert results["node_count"] == 9
        assert results["edge_count"] == 12

        # 指標の存在確認
        required_metrics = [
            "node_count",
            "edge_count",
            "avg_degree",
            "density",
            "alpha_index",
            "beta_index",
            "gamma_index",
            "road_density",
            "avg_circuity",
            "total_length",
            "is_connected",
            "connectivity_ratio"
        ]

        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], int | float | bool)

    def test_empty_graph_handling(self):
        """空グラフの処理テスト"""
        empty_graph = nx.Graph()
        results = self.metrics_calculator.calculate_all_metrics(empty_graph)

        # 空のメトリクスが返されることを確認
        assert results["node_count"] == 0
        assert results["edge_count"] == 0
        assert results["avg_degree"] == 0.0
        assert results["density"] == 0.0

    def test_disconnected_graph_handling(self):
        """非連結グラフの処理テスト"""
        # 2つの分離したコンポーネントを作成
        disconnected_graph = nx.Graph()
        disconnected_graph.add_edge(1, 2, length=100)
        disconnected_graph.add_edge(3, 4, length=100)

        # 座標を追加
        coords = {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (300, 0)}
        for node, (x, y) in coords.items():
            disconnected_graph.nodes[node]["x"] = x
            disconnected_graph.nodes[node]["y"] = y

        # エラーなく処理されることを確認
        results = self.metrics_calculator.calculate_all_metrics(disconnected_graph)

        assert results["node_count"] == 4
        assert results["edge_count"] == 2
        assert results["is_connected"] is False
        assert results["connectivity_ratio"] < 1.0

    def test_basic_metrics(self):
        """基本指標計算テスト"""
        # 単純な線形グラフ
        linear_graph = nx.path_graph(5)
        results = self.metrics_calculator._calculate_basic_metrics(linear_graph)

        assert results["node_count"] == 5
        assert results["edge_count"] == 4
        assert results["avg_degree"] == 1.6  # 4*2/5 = 1.6
        assert results["max_degree"] == 2
        assert results["min_degree"] == 1

    def test_space_syntax_indices(self):
        """Space Syntax指数計算テスト"""
        # 三角形グラフ（1つの回路）
        triangle_graph = nx.cycle_graph(3)
        results = self.metrics_calculator._calculate_space_syntax_indices(triangle_graph)

        assert results["alpha_index"] > 0  # 回路があるのでα指数は正
        assert results["beta_index"] == 1.0  # 3エッジ/3ノード = 1.0
        assert results["gamma_index"] > 0

    def test_connectivity_metrics(self):
        """連結性指標計算テスト"""
        # 完全グラフ
        complete_graph = nx.complete_graph(4)
        results = self.metrics_calculator._calculate_connectivity_metrics(complete_graph)

        assert results["is_connected"] is True
        assert results["connectivity_ratio"] == 1.0
        assert results["largest_component_size"] == 4
        assert results["num_components"] == 1

    def test_density_metrics(self):
        """密度指標計算テスト"""
        # 座標付きのグラフ
        graph = nx.Graph()
        graph.add_node(1, x=0, y=0)
        graph.add_node(2, x=100, y=0)
        graph.add_edge(1, 2, length=100)

        results = self.metrics_calculator._calculate_density_metrics(graph)

        assert results["total_length"] == 100
        assert results["avg_edge_length"] == 100
        assert results["road_density"] >= 0
        assert results["avg_circuity"] >= 1.0

    def test_integration_values(self):
        """Integration値計算テスト"""
        # 線形グラフ
        linear_graph = nx.path_graph(5)
        integration_values = self.metrics_calculator.calculate_integration_values(linear_graph)

        assert len(integration_values) == 5
        # 中央のノードが最も高いIntegration値を持つはず
        center_node = 2
        edge_nodes = [0, 4]
        assert integration_values[center_node] > integration_values[edge_nodes[0]]
        assert integration_values[center_node] > integration_values[edge_nodes[1]]

    def test_choice_values(self):
        """Choice値計算テスト"""
        # スターグラフ（中央ノードが重要）
        star_graph = nx.star_graph(4)
        choice_values = self.metrics_calculator.calculate_choice_values(star_graph)

        assert len(choice_values) == 5
        # 中央ノード（0）が最も高いChoice値を持つはず
        center_node = 0
        assert choice_values[center_node] > 0
        # 周辺ノードのChoice値は0
        for i in range(1, 5):
            assert choice_values[i] == 0.0


class TestConnectivityMetrics:
    """ConnectivityMetrics相当のテストクラス"""

    def test_alpha_index_bounds(self):
        """α指数の境界値テスト"""
        metrics_calc = SpaceSyntaxMetrics()

        # 完全グラフ
        complete_graph = nx.complete_graph(5)
        results = metrics_calc._calculate_space_syntax_indices(complete_graph)
        alpha = results["alpha_index"]
        # α指数は100%を超える場合があるので、0以上であることのみ確認
        assert alpha >= 0

        # 木構造
        tree_graph = nx.path_graph(5)
        results_tree = metrics_calc._calculate_space_syntax_indices(tree_graph)
        alpha_tree = results_tree["alpha_index"]
        assert alpha_tree == 0

    def test_beta_index_calculation(self):
        """β指数の計算テスト"""
        metrics_calc = SpaceSyntaxMetrics()

        # スターグラフ（中心ノードの次数が高い）
        star_graph = nx.star_graph(4)
        results = metrics_calc._calculate_space_syntax_indices(star_graph)
        beta = results["beta_index"]
        # 5ノード、4エッジ → β = 4/5 = 0.8
        assert abs(beta - 0.8) < 0.001

    def test_gamma_index_complete_graph(self):
        """γ指数の完全グラフテスト"""
        metrics_calc = SpaceSyntaxMetrics()

        complete_graph = nx.complete_graph(4)
        results = metrics_calc._calculate_space_syntax_indices(complete_graph)
        gamma = results["gamma_index"]
        # 完全グラフではγ = 100%になるはず
        assert gamma == 1.0  # 正規化されているので1.0


class TestAccessibilityMetrics:
    """AccessibilityMetrics相当のテストクラス"""

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
        metrics_calc = SpaceSyntaxMetrics()
        results = metrics_calc._calculate_density_metrics(self.simple_graph)

        total_length = results["total_length"]
        assert total_length == 200.0  # 100 + 100

    def test_average_shortest_path(self):
        """平均最短距離計算テスト"""
        metrics_calc = SpaceSyntaxMetrics()
        integration_values = metrics_calc.calculate_integration_values(self.simple_graph)

        # Integration値が計算されることを確認
        assert len(integration_values) == 3
        assert all(value >= 0 for value in integration_values.values())


class TestCircuityMetrics:
    """CircuityMetrics相当のテストクラス"""

    def setup_method(self):
        """テスト用ネットワークの準備"""
        # 直線的な道路
        self.straight_graph = nx.Graph()
        self.straight_graph.add_node(1, x=0, y=0)
        self.straight_graph.add_node(2, x=100, y=0)
        self.straight_graph.add_edge(1, 2, length=100)

    def test_circuity_calculation(self):
        """迂回率計算テスト"""
        metrics_calc = SpaceSyntaxMetrics()
        results = metrics_calc._calculate_density_metrics(self.straight_graph)

        circuity = results["avg_circuity"]
        # 迂回率は1.0以上であることを確認
        assert circuity >= 1.0


class TestHelperFunctions:
    """ヘルパー関数のテストクラス"""

    def test_calculate_basic_network_stats(self):
        """基本ネットワーク統計計算テスト"""
        graph = nx.cycle_graph(4)
        stats = calculate_basic_network_stats(graph)

        assert stats["node_count"] == 4
        assert stats["edge_count"] == 4
        assert stats["avg_degree"] == 2.0
        assert stats["density"] > 0

    def test_calculate_space_syntax_indices(self):
        """Space Syntax指数計算テスト"""
        graph = nx.cycle_graph(5)
        indices = calculate_space_syntax_indices(graph)

        assert "alpha_index" in indices
        assert "beta_index" in indices
        assert "gamma_index" in indices
        assert indices["alpha_index"] > 0  # 回路があるので正の値

    def test_empty_graph_stats(self):
        """空グラフの統計テスト"""
        empty_graph = nx.Graph()
        stats = calculate_basic_network_stats(empty_graph)

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["avg_degree"] == 0.0

    def test_small_graph_indices(self):
        """小さなグラフの指数テスト"""
        # 3ノード未満のグラフ
        small_graph = nx.Graph()
        small_graph.add_edge(1, 2)

        indices = calculate_space_syntax_indices(small_graph)

        assert indices["alpha_index"] == 0.0
        assert indices["beta_index"] == 0.0
        assert indices["gamma_index"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
