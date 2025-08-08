"""
ユーティリティ関数のテストモジュール
"""

import networkx as nx
import pytest

from space_syntax_analyzer.utils import (
    calculate_graph_bounds,
    classify_network_type,
    create_bbox_from_center,
    generate_comparison_summary,
    normalize_metrics,
    validate_graph,
)


class TestGraphValidation:
    """グラフ検証機能のテストクラス"""

    def test_valid_graph(self):
        """有効なグラフの検証テスト"""
        graph = nx.Graph()
        graph.add_node(1, x=0, y=0)
        graph.add_node(2, x=1, y=1)
        graph.add_edge(1, 2)

        assert validate_graph(graph) is True

    def test_empty_graph(self):
        """空のグラフの検証テスト"""
        empty_graph = nx.Graph()
        assert validate_graph(empty_graph) is False

    def test_invalid_input(self):
        """不正な入力の検証テスト"""
        assert validate_graph("not_a_graph") is False
        assert validate_graph(None) is False
        assert validate_graph(123) is False


class TestGraphBounds:
    """グラフ境界計算のテストクラス"""

    def test_calculate_bounds(self):
        """境界計算テスト"""
        graph = nx.Graph()
        graph.add_node(1, x=0, y=0)
        graph.add_node(2, x=10, y=5)
        graph.add_node(3, x=-5, y=10)

        bounds = calculate_graph_bounds(graph)
        assert bounds == (-5, 0, 10, 10)  # (min_x, min_y, max_x, max_y)

    def test_no_coordinates(self):
        """座標なしグラフの境界計算テスト"""
        graph = nx.Graph()
        graph.add_node(1)
        graph.add_node(2)

        bounds = calculate_graph_bounds(graph)
        assert bounds is None


class TestMetricsNormalization:
    """指標正規化のテストクラス"""

    def test_normalize_metrics(self):
        """指標正規化テスト"""
        metrics = {
            'alpha_index': 50.0,
            'beta_index': 1.5,
            'gamma_index': 75.0,
            'other_value': 123,
        }

        normalized = normalize_metrics(metrics)

        # 正規化された値が追加されているかチェック
        assert 'alpha_index_normalized' in normalized
        assert 'beta_index_normalized' in normalized
        assert 'gamma_index_normalized' in normalized

        # 正規化値の範囲チェック
        assert 0 <= normalized['alpha_index_normalized'] <= 1
        assert 0 <= normalized['beta_index_normalized'] <= 1
        assert 0 <= normalized['gamma_index_normalized'] <= 1

    def test_custom_reference_values(self):
        """カスタム参照値での正規化テスト"""
        metrics = {'alpha_index': 25.0}
        reference_values = {'alpha_index': 50.0}

        normalized = normalize_metrics(metrics, reference_values)
        assert normalized['alpha_index_normalized'] == 0.5


class TestNetworkClassification:
    """ネットワーク分類のテストクラス"""

    def test_grid_type_classification(self):
        """格子型ネットワーク分類テスト"""
        metrics = {'alpha_index': 35.0, 'beta_index': 1.8, 'gamma_index': 70.0}

        network_type = classify_network_type(metrics)
        assert network_type == "格子型"

    def test_tree_type_classification(self):
        """樹状型ネットワーク分類テスト"""
        metrics = {'alpha_index': 5.0, 'beta_index': 1.1, 'gamma_index': 30.0}

        network_type = classify_network_type(metrics)
        assert network_type == "樹状型"

    def test_radial_type_classification(self):
        """放射型ネットワーク分類テスト"""
        metrics = {'alpha_index': 25.0, 'beta_index': 2.0, 'gamma_index': 55.0}

        network_type = classify_network_type(metrics)
        assert network_type == "放射型"

    def test_irregular_type_classification(self):
        """不定型ネットワーク分類テスト"""
        metrics = {'alpha_index': 15.0, 'beta_index': 1.3, 'gamma_index': 45.0}

        network_type = classify_network_type(metrics)
        assert network_type == "不定型"


class TestComparisonSummary:
    """比較サマリー生成のテストクラス"""

    def test_generate_comparison_summary(self):
        """比較サマリー生成テスト"""
        major_results = {
            'alpha_index': 20.0,
            'avg_shortest_path': 300.0,
            'avg_circuity': 1.5,
        }

        full_results = {
            'alpha_index': 30.0,  # +10の向上
            'avg_shortest_path': 250.0,  # -50の改善
            'avg_circuity': 1.3,  # -0.2の改善
        }

        summary = generate_comparison_summary(major_results, full_results)

        assert 'connectivity' in summary
        assert 'accessibility' in summary
        assert 'circuity' in summary
        assert 'network_type' in summary

        # 改善が反映されているかチェック
        assert "向上" in summary['connectivity']
        assert "向上" in summary['accessibility']
        assert "改善" in summary['circuity']


class TestBboxCreation:
    """境界ボックス作成のテストクラス"""

    def test_create_bbox_from_center(self):
        """中心点からの境界ボックス作成テスト"""
        # 東京駅の座標
        center_lat, center_lon = 35.6812, 139.7671
        distance_km = 1.0

        bbox = create_bbox_from_center(center_lat, center_lon, distance_km)

        # 4つの要素（north, south, east, west）を持つ
        assert len(bbox) == 4
        north, south, east, west = bbox

        # north > south, east > west であることを確認
        assert north > south
        assert east > west

        # 中心座標が範囲内にあることを確認
        assert south < center_lat < north
        assert west < center_lon < east

        # 距離の妥当性チェック（概算）
        lat_diff = north - south
        expected_lat_diff = distance_km * 2 / 111.0  # 緯度1度≈111km
        assert abs(lat_diff - expected_lat_diff) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
