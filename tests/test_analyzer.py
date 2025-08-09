# tests/test_analyzer.py（修正版 - 画像生成処理を排除 v2）
"""
SpaceSyntaxAnalyzerのテストモジュール（修正版 - 画像生成なし v2）
"""

from unittest.mock import MagicMock, patch

# matplotlibのバックエンドを非インタラクティブに設定（画像生成を防ぐ）
import matplotlib
import networkx as nx
import pytest

matplotlib.use("Agg")

from space_syntax_analyzer import SpaceSyntaxAnalyzer
from space_syntax_analyzer.core.metrics import SpaceSyntaxMetrics


class TestSpaceSyntaxAnalyzer:
    """SpaceSyntaxAnalyzerのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される準備処理"""
        self.analyzer = SpaceSyntaxAnalyzer()

    def test_analyzer_initialization(self):
        """アナライザーの初期化テスト"""
        analyzer = SpaceSyntaxAnalyzer(width_threshold=5.0, network_type="walk")
        assert analyzer.width_threshold == 5.0
        assert analyzer.network_type == "walk"
        assert hasattr(analyzer, "network_manager")
        assert hasattr(analyzer, "metrics")
        assert hasattr(analyzer, "visualizer")

    def test_get_empty_metrics(self):
        """空の指標取得テスト"""
        empty_metrics = self.analyzer._get_empty_metrics()

        required_keys = [
            "node_count", "edge_count", "avg_degree", "alpha_index",
            "beta_index", "gamma_index", "road_density", "avg_circuity"
        ]

        for key in required_keys:
            assert key in empty_metrics
            assert isinstance(empty_metrics[key], int | float)

    def test_create_empty_result(self):
        """空の結果作成テスト"""
        # return_networks = False の場合
        result = self.analyzer._create_empty_result(False, "テストエラー")

        assert "major_network" in result
        assert "full_network" in result
        assert "metadata" in result
        assert result["metadata"]["error_message"] == "テストエラー"

        # return_networks = True の場合
        result, networks = self.analyzer._create_empty_result(True, "テストエラー")
        assert networks == (None, None)

    @patch("space_syntax_analyzer.core.analyzer.SpaceSyntaxAnalyzer._get_networks")
    def test_analyze_place_success(self, mock_get_networks):
        """地域分析成功テスト"""
        # モックネットワークの作成
        mock_major_net = nx.Graph()
        mock_major_net.add_edges_from([(1, 2), (2, 3)])
        # 座標を追加
        for i, node in enumerate(mock_major_net.nodes()):
            mock_major_net.nodes[node]["x"] = i * 100
            mock_major_net.nodes[node]["y"] = 0

        mock_full_net = nx.Graph()
        mock_full_net.add_edges_from([(1, 2), (2, 3), (3, 4)])
        # 座標を追加
        for i, node in enumerate(mock_full_net.nodes()):
            mock_full_net.nodes[node]["x"] = i * 100
            mock_full_net.nodes[node]["y"] = 0

        mock_get_networks.return_value = (mock_major_net, mock_full_net)

        # 分析実行
        results = self.analyzer.analyze_place("Test Location")

        # 結果の確認
        assert "major_network" in results
        assert "full_network" in results
        assert "metadata" in results
        assert results["metadata"]["analysis_status"] == "success"

    @patch("space_syntax_analyzer.core.analyzer.SpaceSyntaxAnalyzer._get_networks")
    def test_analyze_place_failure(self, mock_get_networks):
        """地域分析失敗テスト"""
        mock_get_networks.return_value = (None, None)

        results = self.analyzer.analyze_place("Invalid Location")

        assert results["metadata"]["analysis_status"] == "failed"
        assert "error_message" in results["metadata"]

    def test_analyze_network(self):
        """ネットワーク分析テスト"""
        # テスト用グラフ作成
        test_graph = nx.Graph()
        test_graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

        # 座標の追加
        coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}
        for node, (x, y) in coords.items():
            test_graph.nodes[node]["x"] = x
            test_graph.nodes[node]["y"] = y

        results = self.analyzer._analyze_network(test_graph, "テストネットワーク")

        assert results["network_name"] == "テストネットワーク"
        assert results["analysis_status"] == "success"
        assert results["node_count"] == 4
        assert results["edge_count"] == 3

    def test_analyze_empty_network(self):
        """空ネットワーク分析テスト"""
        empty_graph = nx.Graph()
        results = self.analyzer._analyze_network(empty_graph, "空ネットワーク")

        assert results["network_name"] == "空ネットワーク"
        assert results["node_count"] == 0
        assert results["edge_count"] == 0

    def test_get_network_both(self):
        """ネットワーク取得テスト（両方）"""
        with patch.object(self.analyzer, "_get_networks") as mock_get_networks:
            mock_major = nx.Graph()
            mock_full = nx.Graph()
            mock_get_networks.return_value = (mock_major, mock_full)

            result = self.analyzer.get_network("Test Location", "both")
            assert result == (mock_major, mock_full)

    def test_get_network_major_only(self):
        """ネットワーク取得テスト（主要道路のみ）"""
        with patch.object(self.analyzer, "_get_networks") as mock_get_networks:
            mock_major = nx.Graph()
            mock_full = nx.Graph()
            mock_get_networks.return_value = (mock_major, mock_full)

            result = self.analyzer.get_network("Test Location", "major")
            assert result == mock_major

    def test_get_network_full_only(self):
        """ネットワーク取得テスト（全道路のみ）"""
        with patch.object(self.analyzer, "_get_networks") as mock_get_networks:
            mock_major = nx.Graph()
            mock_full = nx.Graph()
            mock_get_networks.return_value = (mock_major, mock_full)

            result = self.analyzer.get_network("Test Location", "full")
            assert result == mock_full

    def test_get_network_invalid_selection(self):
        """不正なネットワーク選択テスト"""
        # get_networkメソッドは実際にはエラーをログに出力してNoneを返すため、
        # ValueErrorではなくNoneが返されることを確認
        result = self.analyzer.get_network("Test Location", "invalid")
        assert result is None

    def test_generate_report(self):
        """レポート生成テスト"""
        # テスト用の結果データ
        results = {
            "metadata": {
                "query": "Test Location",
                "network_type": "drive",
                "analysis_status": "success"
            },
            "major_network": {
                "node_count": 10,
                "edge_count": 12,
                "alpha_index": 25.0,
                "beta_index": 1.2,
                "avg_circuity": 1.3,
                "density": 0.15
            },
            "full_network": {
                "node_count": 20,
                "edge_count": 25,
                "alpha_index": 30.0,
                "beta_index": 1.25,
                "avg_circuity": 1.2,
                "density": 0.12
            }
        }

        report = self.analyzer.generate_report(results, "テスト地域")

        assert "テスト地域" in report
        assert "Test Location" in report
        assert "ノード数: 10" in report
        assert "α指数: 25.0%" in report
        assert "主要道路ネットワーク" in report
        assert "全道路ネットワーク" in report

    def test_generate_report_incomplete_data(self):
        """不完全なデータでのレポート生成テスト"""
        incomplete_results = {
            "metadata": {"query": "Test"},
            "major_network": None,
            "full_network": None
        }

        report = self.analyzer.generate_report(incomplete_results)
        assert isinstance(report, str)
        assert "Test" in report

    def test_format_network_section(self):
        """ネットワークセクションフォーマットテスト"""
        network_data = {
            "node_count": 15,
            "edge_count": 18,
            "avg_degree": 2.4,
            "density": 0.2,
            "alpha_index": 30.0,
            "beta_index": 1.2,
            "gamma_index": 0.8,
            "avg_circuity": 1.1,
            "road_density": 25.5
        }

        lines = self.analyzer._format_network_section("テストネットワーク", network_data)

        assert "【テストネットワーク】" in lines
        assert any("ノード数: 15" in line for line in lines)
        assert any("α指数: 30.0%" in line for line in lines)

    def test_format_comparison_section(self):
        """比較セクションフォーマットテスト"""
        major_data = {"node_count": 10, "edge_count": 12, "avg_degree": 2.4}
        full_data = {"node_count": 20, "edge_count": 25, "avg_degree": 2.5}

        lines = self.analyzer._format_comparison_section(major_data, full_data)

        assert "【ネットワーク比較】" in lines
        assert any("50.0%" in line for line in lines)  # 10/20 * 100

    def test_export_results_error_handling(self):
        """結果出力エラーハンドリングテスト"""
        invalid_results = {"invalid": "data"}

        # 不正なフォーマットでエラーが処理されることを確認
        success = self.analyzer.export_results(invalid_results, "test.invalid", "invalid_format")
        assert success is False

    def test_results_to_dataframe(self):
        """結果をDataFrameに変換するテスト"""
        results = {
            "major_network": {
                "node_count": 10,
                "edge_count": 12,
                "alpha_index": 25.0
            },
            "full_network": {
                "node_count": 20,
                "edge_count": 25,
                "alpha_index": 30.0
            }
        }

        df = self.analyzer._results_to_dataframe(results)

        assert len(df) == 2
        assert "network_type" in df.columns
        assert "node_count" in df.columns
        assert df.iloc[0]["network_type"] == "major_network"
        assert df.iloc[1]["network_type"] == "full_network"

    @patch("matplotlib.pyplot.show")  # plt.show()をモック化して実際の表示を防ぐ
    @patch("matplotlib.pyplot.savefig")  # savefigもモック化
    @patch("matplotlib.pyplot.subplots")  # subplotsもモック化
    @patch("matplotlib.pyplot.tight_layout")  # tight_layoutもモック化
    def test_visualize_success(self, mock_tight_layout, mock_subplots, mock_savefig, mock_show):
        """可視化成功テスト（画像生成なし）"""
        # モックのfigとaxesを作成
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        mock_major = nx.Graph()
        mock_full = nx.Graph()
        results = {"metadata": {"query": "Test"}}

        # visualizer自体にplot_network_comparisonが存在しないため、
        # 最初から_basic_visualizationが呼ばれることを前提とする
        success = self.analyzer.visualize(mock_major, mock_full, results)

        # 成功することを確認
        assert success is True

        # matplotlibの関数が呼ばれたことを確認
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()  # save_path=Noneなので呼ばれない

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_visualize_failure(self, mock_subplots, mock_savefig, mock_show):
        """可視化失敗テスト（画像生成なし）"""
        mock_major = nx.Graph()
        mock_full = nx.Graph()
        results = {"metadata": {"query": "Test"}}

        # subplotsで例外が発生するようにモック化
        mock_subplots.side_effect = Exception("Test error")

        success = self.analyzer.visualize(mock_major, mock_full, results)
        assert success is False

        # 例外が発生した場合、show/savefigは呼ばれない
        mock_show.assert_not_called()
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_basic_visualization_with_save(self, mock_subplots, mock_savefig, mock_show):
        """基本可視化のファイル保存テスト（画像生成なし）"""
        # モックのfigとaxesを作成
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_axes.flat = [MagicMock() for _ in range(4)]  # 2x2のsubplot
        mock_subplots.return_value = (mock_fig, mock_axes)

        mock_major = nx.Graph()
        mock_full = nx.Graph()
        results = {"metadata": {"query": "Test"}}
        save_path = "test_output.png"

        # save_pathを指定して実行
        success = self.analyzer.visualize(mock_major, mock_full, results, save_path=save_path)

        assert success is True

        # ファイル保存が呼ばれることを確認
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
        mock_show.assert_called_once()

    def test_visualize_with_empty_networks(self):
        """空のネットワークでの可視化テスト"""
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = MagicMock()
            mock_axes.flat = [MagicMock() for _ in range(4)]
            mock_subplots.return_value = (mock_fig, mock_axes)

            # 空のネットワーク
            empty_major = None
            empty_full = None
            results = {"metadata": {"query": "Empty Test"}}

            success = self.analyzer.visualize(empty_major, empty_full, results)

            # 空のネットワークでも正常に処理されることを確認
            assert success is True
            mock_subplots.assert_called_once()


class TestSpaceSyntaxMetricsIntegration:
    """SpaceSyntaxMetricsとの統合テストクラス"""

    def test_metrics_integration(self):
        """メトリクス統合テスト"""
        metrics_calc = SpaceSyntaxMetrics()

        # テスト用グラフ
        test_graph = nx.cycle_graph(5)
        # 座標を追加
        for i, node in enumerate(test_graph.nodes()):
            test_graph.nodes[node]["x"] = i * 100
            test_graph.nodes[node]["y"] = 0

        results = metrics_calc.calculate_all_metrics(test_graph)

        # 基本的な結果の確認
        assert results["node_count"] == 5
        assert results["edge_count"] == 5
        assert results["alpha_index"] > 0  # 回路があるので正の値

    def test_analyzer_metrics_usage(self):
        """アナライザーでのメトリクス使用テスト"""
        analyzer = SpaceSyntaxAnalyzer()

        # テスト用グラフ
        test_graph = nx.Graph()
        test_graph.add_edges_from([(1, 2), (2, 3), (3, 1)])  # 三角形
        # 座標を追加
        coords = {1: (0, 0), 2: (100, 0), 3: (50, 100)}
        for node, (x, y) in coords.items():
            test_graph.nodes[node]["x"] = x
            test_graph.nodes[node]["y"] = y

        results = analyzer._analyze_network(test_graph, "三角形")

        assert results["node_count"] == 3
        assert results["edge_count"] == 3
        assert results["analysis_status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
