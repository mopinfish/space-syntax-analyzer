"""
SpaceSyntaxAnalyzerの軽量テスト（外部依存関係なし）
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports_without_external_deps():
    """外部依存関係なしでのインポートテスト"""
    # 外部ライブラリをモック化
    mock_modules = [
        'shapely', 'shapely.geometry', 'osmnx', 'geopandas', 
        'matplotlib', 'matplotlib.pyplot', 'pandas'
    ]
    
    for module_name in mock_modules:
        sys.modules[module_name] = MagicMock()
    
    try:
        # メインモジュールのインポートテスト
        # 基本的な機能テスト
        import networkx as nx

        from space_syntax_analyzer.core.metrics import ConnectivityMetrics
        from space_syntax_analyzer.utils.helpers import (
            normalize_metrics,
            validate_graph,
        )

        # テスト用の簡単なグラフ（三角形）
        graph = nx.Graph()
        graph.add_node(1, x=0, y=0)
        graph.add_node(2, x=1, y=0)
        graph.add_node(3, x=0.5, y=1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)
        
        print(f"  テストグラフ: 三角形 (ノード={graph.number_of_nodes()}, エッジ={graph.number_of_edges()})")
        
        # ConnectivityMetricsのテスト
        print(f"  グラフ情報: ノード数={graph.number_of_nodes()}, エッジ数={graph.number_of_edges()}")
        
        mu = ConnectivityMetrics.calculate_mu_index(graph)
        print(f"  回路指数(μ): {mu}")
        assert mu == 1  # 三角形なので1つの回路
        
        alpha = ConnectivityMetrics.calculate_alpha_index(graph)
        print(f"  α指数: {alpha:.2f}%")
        assert alpha > 0
        
        beta = ConnectivityMetrics.calculate_beta_index(graph)
        print(f"  β指数: {beta:.2f}")
        expected_beta = graph.number_of_edges() / graph.number_of_nodes()
        print(f"  期待値: {expected_beta:.2f}")
        assert abs(beta - expected_beta) < 0.001
        
        gamma = ConnectivityMetrics.calculate_gamma_index(graph)
        print(f"  γ指数: {gamma:.2f}%")
        assert gamma > 0
        
        # 追加テスト：線形グラフ
        print("\n  追加テスト: 線形グラフ")
        linear_graph = nx.path_graph(4)  # 1-2-3-4の線形グラフ
        for i, node in enumerate(linear_graph.nodes()):
            linear_graph.nodes[node]['x'] = i
            linear_graph.nodes[node]['y'] = 0
            
        print(f"  線形グラフ: ノード={linear_graph.number_of_nodes()}, エッジ={linear_graph.number_of_edges()}")
        
        mu_linear = ConnectivityMetrics.calculate_mu_index(linear_graph)
        beta_linear = ConnectivityMetrics.calculate_beta_index(linear_graph)
        print(f"  線形グラフ μ: {mu_linear} (期待値: 0)")
        print(f"  線形グラフ β: {beta_linear:.2f} (期待値: {linear_graph.number_of_edges()/linear_graph.number_of_nodes():.2f})")
        
        assert mu_linear == 0  # 線形グラフは回路なし
        
        # ユーティリティ関数のテスト
        assert validate_graph(graph) is True
        
        metrics = {'alpha_index': 50.0, 'beta_index': 1.5}
        normalized = normalize_metrics(metrics)
        assert 'alpha_index_normalized' in normalized
        
        print("✅ 軽量テスト完了：基本機能は正常に動作しています")
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports_without_external_deps()
    sys.exit(0 if success else 1)