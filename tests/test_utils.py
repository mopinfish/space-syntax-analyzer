"""
ユーティリティ関数のテストモジュール（修正版）
"""

import networkx as nx
import pytest

from space_syntax_analyzer.core.network import create_bbox_from_center
from space_syntax_analyzer.utils.helpers import (
    calculate_bbox_area,
    check_osmnx_version,
    create_analysis_summary,
    create_network_comparison_report,
    debug_network_info,
    estimate_processing_time,
    format_coordinates,
    generate_comparison_summary,
    setup_logging,
    validate_bbox,
)


class TestBboxValidation:
    """境界ボックス検証機能のテストクラス"""

    def test_valid_bbox(self):
        """有効なbboxの検証テスト"""
        # 東京駅周辺の有効なbbox
        bbox = (139.7, 35.67, 139.8, 35.69)
        assert validate_bbox(bbox) is True

    def test_invalid_bbox_order(self):
        """不正な順序のbboxの検証テスト"""
        # left >= right の場合
        bbox = (139.8, 35.67, 139.7, 35.69)
        assert validate_bbox(bbox) is False

        # bottom >= top の場合
        bbox = (139.7, 35.69, 139.8, 35.67)
        assert validate_bbox(bbox) is False

    def test_invalid_bbox_range(self):
        """範囲外のbboxの検証テスト"""
        # 経度が範囲外
        bbox = (-200, 35.67, 139.8, 35.69)
        assert validate_bbox(bbox) is False

        # 緯度が範囲外
        bbox = (139.7, -100, 139.8, 35.69)
        assert validate_bbox(bbox) is False

    def test_too_large_bbox(self):
        """大きすぎるbboxの検証テスト"""
        # 5度以上の範囲
        bbox = (139.7, 35.67, 145.0, 41.0)
        assert validate_bbox(bbox) is False

    def test_invalid_input_type(self):
        """不正な入力タイプの検証テスト"""
        assert validate_bbox("not_a_bbox") is False
        assert validate_bbox(None) is False
        assert validate_bbox([139.7, 35.67, 139.8]) is False  # 要素数不足


class TestCoordinateFormatting:
    """座標フォーマット機能のテストクラス"""

    def test_format_positive_coordinates(self):
        """正の座標のフォーマットテスト"""
        result = format_coordinates(35.6812, 139.7671)
        assert "N" in result
        assert "E" in result
        assert "35.6812" in result
        assert "139.7671" in result

    def test_format_negative_coordinates(self):
        """負の座標のフォーマットテスト"""
        result = format_coordinates(-35.6812, -139.7671)
        assert "S" in result
        assert "W" in result

    def test_format_mixed_coordinates(self):
        """正負混合の座標のフォーマットテスト"""
        result = format_coordinates(35.6812, -139.7671)
        assert "N" in result
        assert "W" in result

    def test_precision_parameter(self):
        """精度パラメータのテスト"""
        result = format_coordinates(35.6812, 139.7671, precision=2)
        assert "35.68" in result
        assert "139.77" in result


class TestBboxArea:
    """境界ボックス面積計算のテストクラス"""

    def test_calculate_small_bbox_area(self):
        """小さなbboxの面積計算テスト"""
        # 1度x1度のbbox（東京付近）
        bbox = (139.0, 35.0, 140.0, 36.0)
        area = calculate_bbox_area(bbox)

        # 大まかに111km x 111km = 約12,321 km²
        assert 10000 < area < 15000

    def test_zero_area_bbox(self):
        """面積ゼロのbboxのテスト"""
        # 同じ座標のbbox
        bbox = (139.0, 35.0, 139.0, 35.0)
        area = calculate_bbox_area(bbox)
        assert area == 0.0

    def test_invalid_bbox_area(self):
        """不正なbboxの面積計算テスト"""
        # 不正な形式のbbox
        area = calculate_bbox_area("invalid")
        assert area == 0.0


class TestComparisonSummary:
    """比較サマリー生成のテストクラス"""

    def test_generate_comparison_summary(self):
        """比較サマリー生成テスト"""
        major_results = {
            'node_count': 100,
            'edge_count': 120,
            'avg_degree': 2.4,
            'alpha_index': 20.0,
            'beta_index': 1.2
        }

        full_results = {
            'node_count': 300,
            'edge_count': 400,
            'avg_degree': 2.7,
            'alpha_index': 30.0,
            'beta_index': 1.3
        }

        summary = generate_comparison_summary(major_results, full_results)

        # 必要なキーが存在することを確認
        assert '主要道路ノード数' in summary
        assert '全道路ノード数' in summary
        assert '主要道路比率' in summary
        assert '主要道路エッジ数' in summary
        assert '全道路エッジ数' in summary

        # 計算結果の確認
        assert summary['主要道路ノード数'] == 100
        assert summary['全道路ノード数'] == 300
        assert "33.3%" in summary['主要道路比率']

    def test_empty_results_summary(self):
        """空の結果の比較サマリーテスト"""
        major_results = {}
        full_results = {}

        summary = generate_comparison_summary(major_results, full_results)

        # エラーが発生しないことを確認
        assert isinstance(summary, dict)
        assert summary['主要道路ノード数'] == 0
        assert summary['全道路ノード数'] == 0


class TestBboxCreation:
    """境界ボックス作成のテストクラス"""

    def test_create_bbox_from_center(self):
        """中心点からの境界ボックス作成テスト"""
        # 東京駅の座標
        center_lat, center_lon = 35.6812, 139.7671
        distance_km = 1.0

        bbox = create_bbox_from_center(center_lat, center_lon, distance_km)

        # 4つの要素（left, bottom, right, top）を持つ
        assert len(bbox) == 4
        left, bottom, right, top = bbox

        # right > left, top > bottom であることを確認
        assert right > left
        assert top > bottom

        # 中心座標が範囲内にあることを確認
        assert bottom < center_lat < top
        assert left < center_lon < right

    def test_small_distance_bbox(self):
        """小さな距離でのbbox作成テスト"""
        center_lat, center_lon = 35.6812, 139.7671
        distance_km = 0.1  # 100m

        bbox = create_bbox_from_center(center_lat, center_lon, distance_km)
        left, bottom, right, top = bbox

        # 小さな範囲であることを確認
        assert (top - bottom) < 0.01
        assert (right - left) < 0.01

    def test_large_distance_bbox(self):
        """大きな距離でのbbox作成テスト"""
        center_lat, center_lon = 35.6812, 139.7671
        distance_km = 10.0  # 10km

        bbox = create_bbox_from_center(center_lat, center_lon, distance_km)
        left, bottom, right, top = bbox

        # 大きな範囲であることを確認
        assert (top - bottom) > 0.1
        assert (right - left) > 0.1


class TestAnalysisSummary:
    """分析サマリー作成のテストクラス"""

    def test_create_analysis_summary(self):
        """分析サマリー作成テスト"""
        results = {
            'metadata': {
                'query': 'Test Location',
                'network_type': 'drive',
                'analysis_status': 'success'
            },
            'major_network': {
                'node_count': 100,
                'edge_count': 120,
                'alpha_index': 25.0,
                'connectivity_ratio': 0.95
            },
            'full_network': {
                'node_count': 300,
                'edge_count': 400,
                'alpha_index': 35.0,
                'connectivity_ratio': 0.98
            }
        }

        summary = create_analysis_summary(results)

        # 基本情報の確認
        assert summary['query'] == 'Test Location'
        assert summary['network_type'] == 'drive'
        assert summary['analysis_status'] == 'success'

        # ネットワーク情報の確認
        assert summary['major_nodes'] == 100
        assert summary['full_nodes'] == 300
        assert summary['major_ratio'] == 100/300*100

    def test_incomplete_results_summary(self):
        """不完全な結果の分析サマリーテスト"""
        results = {
            'metadata': {'query': 'Test'},
            'major_network': None,
            'full_network': None
        }

        summary = create_analysis_summary(results)

        # エラーが発生しないことを確認
        assert isinstance(summary, dict)
        assert summary['query'] == 'Test'


class TestVersionCheck:
    """バージョンチェック機能のテストクラス"""

    def test_check_osmnx_version(self):
        """OSMnxバージョンチェックテスト"""
        version_info = check_osmnx_version()

        # 必要なキーが存在することを確認
        assert isinstance(version_info, dict)

        # エラーまたは正常な情報のいずれかが返されることを確認
        if 'error' in version_info:
            assert isinstance(version_info['error'], str)
        else:
            assert 'osmnx' in version_info
            assert 'networkx' in version_info


class TestProcessingTimeEstimation:
    """処理時間見積もり機能のテストクラス"""

    def test_estimate_processing_time_small(self):
        """小さなエリアの処理時間見積もりテスト"""
        small_bbox = (139.7, 35.67, 139.71, 35.68)  # 約1km²未満
        estimate = estimate_processing_time(small_bbox)
        assert "30秒" in estimate

    def test_estimate_processing_time_large(self):
        """大きなエリアの処理時間見積もりテスト"""
        large_bbox = (139.0, 35.0, 140.0, 36.0)  # 約1度×1度
        estimate = estimate_processing_time(large_bbox)
        assert "10分" in estimate or "以上" in estimate

    def test_invalid_bbox_estimation(self):
        """不正なbboxの処理時間見積もりテスト"""
        estimate = estimate_processing_time("invalid")
        assert estimate == "不明"


class TestNetworkComparisonReport:
    """ネットワーク比較レポート機能のテストクラス"""

    def test_create_network_comparison_report_both_networks(self):
        """両方のネットワークがある場合の比較レポートテスト"""
        # モックネットワーク作成
        major_net = nx.MultiDiGraph()
        major_net.add_nodes_from(range(10))
        major_net.add_edges_from([(i, i+1) for i in range(9)])

        full_net = nx.MultiDiGraph()
        full_net.add_nodes_from(range(20))
        full_net.add_edges_from([(i, i+1) for i in range(19)])

        report = create_network_comparison_report(major_net, full_net)

        assert "ネットワーク比較レポート" in report
        assert "10 ノード" in report
        assert "20 ノード" in report
        assert "主要道路比率" in report

    def test_create_network_comparison_report_no_networks(self):
        """ネットワークがない場合の比較レポートテスト"""
        report = create_network_comparison_report(None, None)

        assert "ネットワーク比較レポート" in report
        assert "両方のネットワークが取得できませんでした" in report

    def test_create_network_comparison_report_partial(self):
        """片方のネットワークのみの比較レポートテスト"""
        full_net = nx.MultiDiGraph()
        full_net.add_nodes_from(range(5))

        report = create_network_comparison_report(None, full_net)

        assert "ネットワーク比較レポート" in report
        assert "5 ノード" in report


class TestDebugNetworkInfo:
    """ネットワークデバッグ情報機能のテストクラス"""

    def test_debug_network_info_with_network(self, capsys):
        """ネットワークありのデバッグ情報テスト"""
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(5))
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        debug_network_info(G, "テストネットワーク")

        captured = capsys.readouterr()
        assert "テストネットワーク デバッグ情報:" in captured.out
        assert "5" in captured.out  # ノード数

    def test_debug_network_info_none(self, capsys):
        """Noneネットワークのデバッグ情報テスト"""
        debug_network_info(None, "空ネットワーク")

        captured = capsys.readouterr()
        assert "空ネットワーク: None" in captured.out

    def test_debug_network_info_empty(self, capsys):
        """空ネットワークのデバッグ情報テスト"""
        G = nx.MultiDiGraph()
        debug_network_info(G, "空のグラフ")

        captured = capsys.readouterr()
        assert "空のグラフ デバッグ情報:" in captured.out
        assert "ノード数: 0" in captured.out


class TestHelperFunctions:
    """その他のヘルパー関数のテストクラス"""

    def test_setup_logging(self):
        """ロギングセットアップテスト"""
        logger = setup_logging("DEBUG")

        # ロガーが返されることを確認
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'error')

    def test_setup_logging_invalid_level(self):
        """不正なログレベルでのセットアップテスト"""
        # 不正なレベルでもエラーが出ないことを確認
        logger = setup_logging("INVALID")
        assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
