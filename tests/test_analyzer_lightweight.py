"""
軽量テスト用モジュール（修正版）
外部依存関係を最小限にした基本機能のテスト
"""

import sys
from unittest.mock import MagicMock


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

        # 実際に存在するクラスをインポート
        from space_syntax_analyzer.core.metrics import SpaceSyntaxMetrics
        from space_syntax_analyzer.utils.helpers import (
            format_coordinates,
            validate_bbox,
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

        # SpaceSyntaxMetricsのテスト
        metrics_calc = SpaceSyntaxMetrics()

        print(f"  グラフ情報: ノード数={graph.number_of_nodes()}, エッジ数={graph.number_of_edges()}")

        # Space Syntax指標のテスト
        space_syntax_results = metrics_calc._calculate_space_syntax_indices(graph)

        alpha = space_syntax_results['alpha_index']
        beta = space_syntax_results['beta_index']
        gamma = space_syntax_results['gamma_index']

        print(f"  α指数: {alpha:.2f}%")
        print(f"  β指数: {beta:.2f}")
        print(f"  γ指数: {gamma:.2f}")

        assert alpha > 0  # 三角形なので回路がある
        assert beta == 1.0  # 3エッジ/3ノード = 1.0
        assert gamma > 0

        # 追加テスト：線形グラフ
        print("\n  追加テスト: 線形グラフ")
        linear_graph = nx.path_graph(4)  # 1-2-3-4の線形グラフ
        for i, node in enumerate(linear_graph.nodes()):
            linear_graph.nodes[node]['x'] = i
            linear_graph.nodes[node]['y'] = 0

        print(f"  線形グラフ: ノード={linear_graph.number_of_nodes()}, エッジ={linear_graph.number_of_edges()}")

        linear_results = metrics_calc._calculate_space_syntax_indices(linear_graph)
        alpha_linear = linear_results['alpha_index']
        beta_linear = linear_results['beta_index']

        print(f"  線形グラフ α: {alpha_linear} (期待値: 0)")
        print(f"  線形グラフ β: {beta_linear:.2f} (期待値: {linear_graph.number_of_edges()/linear_graph.number_of_nodes():.2f})")

        assert alpha_linear == 0  # 線形グラフは回路なし

        # ユーティリティ関数のテスト
        test_bbox = (139.7, 35.67, 139.8, 35.69)
        assert validate_bbox(test_bbox) is True

        # 座標フォーマットテスト
        formatted = format_coordinates(35.6812, 139.7671)
        assert "N" in formatted and "E" in formatted

        print("✅ 軽量テスト完了：基本機能は正常に動作しています")


    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError("テストが失敗しました") from e


def test_basic_network_analysis():
    """基本的なネットワーク分析テスト"""
    try:
        import networkx as nx

        from space_syntax_analyzer.core.metrics import SpaceSyntaxMetrics

        # テスト用グラフ作成
        G = nx.cycle_graph(5)  # 5ノードの環状グラフ

        # 座標追加
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['x'] = i
            G.nodes[node]['y'] = 0

        metrics_calc = SpaceSyntaxMetrics()
        results = metrics_calc.calculate_all_metrics(G)

        # 基本チェック
        assert results['node_count'] == 5
        assert results['edge_count'] == 5
        assert results['alpha_index'] > 0  # 環状なので回路がある

        print("✅ 基本ネットワーク分析テスト完了")

    except Exception as e:
        print(f"❌ 基本ネットワーク分析テストエラー: {e}")
        raise


if __name__ == "__main__":
    test_imports_without_external_deps()
    test_basic_network_analysis()
    print("すべての軽量テストが完了しました")
