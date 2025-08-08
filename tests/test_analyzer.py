"""
SpaceSyntaxAnalyzerのテストモジュール
"""

import pytest
import networkx as nx
from space_syntax_analyzer import SpaceSyntaxAnalyzer
from space_syntax_analyzer.core.metrics import ConnectivityMetrics, AccessibilityMetrics


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
        assert analyzer.crs == "EPSG:4326"
        
    def test_empty_graph_analysis(self):
        """空のグラフに対する分析テスト"""
        empty_graph = nx.Graph()
        results = self.analyzer.analyze(empty_graph)
        
        assert results["major_network"]["nodes"] == 0
        assert results["major_network"]["edges"] == 0
        
    def test_simple_graph_metrics(self):
        """シンプルなグラフに対するメトリクス計算テスト"""
        # 三角形のグラフを作成
        graph = nx.Graph()
        graph.add_node(1, x=0, y=0)
        graph.add_node(2, x=1, y=0) 
        graph.add_node(3, x=0.5, y=1)
        graph.add_edge(1, 2, length=100)
        graph.add_edge(2, 3, length=100)
        graph.add_edge(3, 1, length=100)
        
        results = self.analyzer.analyze(graph, area_ha=1.0)
        metrics = results["major_network"]
        
        assert metrics["nodes"] == 3
        assert metrics["edges"] == 3
        assert metrics["mu_index"] == 1  # 1つの回路
        assert metrics["alpha_index"] > 0
        
    def test_generate_report(self):
        """レポート生成テスト"""
        # テスト用の結果データ
        results = {
            "major_network": {
                "nodes": 10,
                "edges": 12,
                "alpha_index": 25.0,
                "beta_index": 1.2,
                "avg_circuity": 1.3
            }
        }
        
        report = self.analyzer.generate_report(results, "テスト地域")
        
        assert "テスト地域" in report
        assert "ノード数: 10" in report
        assert "α指数: 25.0%" in report


class TestConnectivityMetrics:
    """ConnectivityMetricsのテストクラス"""
    
    def test_mu_index_calculation(self):
        """回路指数計算テスト"""
        # 単純な回路グラフ
        graph = nx.cycle_graph(4)
        mu = ConnectivityMetrics.calculate_mu_index(graph)
        assert mu == 1  # 4ノード、4エッジ、1成分 → 4-4+1=1
        
    def test_alpha_index_calculation(self):
        """α指数計算テスト"""
        # 完全グラフ
        graph = nx.complete_graph(4)
        alpha = ConnectivityMetrics.calculate_alpha_index(graph)
        assert alpha > 0
        assert alpha <= 100
        
    def test_beta_index_calculation(self):
        """β指数計算テスト"""
        graph = nx.path_graph(3)  # 線形グラフ
        beta = ConnectivityMetrics.calculate_beta_index(graph)
        assert beta == 2.0 / 3.0  # 2エッジ / 3ノード
        
    def test_gamma_index_calculation(self):
        """γ指数計算テスト"""
        graph = nx.complete_graph(4)
        gamma = ConnectivityMetrics.calculate_gamma_index(graph)
        assert gamma > 0
        assert gamma <= 100


class TestAccessibilityMetrics:
    """AccessibilityMetricsのテストクラス"""
    
    def test_road_density_calculation(self):
        """道路密度計算テスト"""
        graph = nx.Graph()
        graph.add_edge(1, 2, length=100)
        graph.add_edge(2, 3, length=200)
        
        density = AccessibilityMetrics.calculate_road_density(graph, area_ha=1.0)
        assert density == 300.0  # 300m / 1ha
        
    def test_intersection_density_calculation(self):
        """交差点密度計算テスト"""
        # T字路のグラフ
        graph = nx.Graph()
        graph.add_edge(1, 2)  # 中央ノード2は次数3
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        
        density = AccessibilityMetrics.calculate_intersection_density(graph, area_ha=2.0)
        assert density == 0.5  # 1交差点 / 2ha
        

if __name__ == "__main__":
    pytest.main([__file__])