#!/usr/bin/env python3
"""
デバッグ用テスト - 実際の値を確認
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_connectivity_metrics():
    """ConnectivityMetricsの実際の計算値を確認"""
    print("🔍 ConnectivityMetrics デバッグ")
    print("=" * 40)
    
    try:
        import networkx as nx
        from space_syntax_analyzer.core.metrics import ConnectivityMetrics
        
        # 三角形グラフ
        triangle = nx.Graph()
        triangle.add_edge(1, 2)
        triangle.add_edge(2, 3) 
        triangle.add_edge(3, 1)
        
        print("📐 三角形グラフ:")
        print(f"  ノード数: {triangle.number_of_nodes()}")
        print(f"  エッジ数: {triangle.number_of_edges()}")
        print(f"  連結成分数: {nx.number_connected_components(triangle)}")
        
        # 各指標を計算
        mu = ConnectivityMetrics.calculate_mu_index(triangle)
        alpha = ConnectivityMetrics.calculate_alpha_index(triangle)
        beta = ConnectivityMetrics.calculate_beta_index(triangle)
        gamma = ConnectivityMetrics.calculate_gamma_index(triangle)
        
        print(f"\n📊 計算結果:")
        print(f"  μ指数: {mu}")
        print(f"  α指数: {alpha:.4f}%")
        print(f"  β指数: {beta:.4f}")
        print(f"  γ指数: {gamma:.4f}%")
        
        # 手動計算と比較
        print(f"\n🧮 手動計算:")
        manual_mu = triangle.number_of_edges() - triangle.number_of_nodes() + nx.number_connected_components(triangle)
        print(f"  μ = e - ν + p = {triangle.number_of_edges()} - {triangle.number_of_nodes()} + {nx.number_connected_components(triangle)} = {manual_mu}")
        
        manual_beta = triangle.number_of_edges() / triangle.number_of_nodes()
        print(f"  β = e / ν = {triangle.number_of_edges()} / {triangle.number_of_nodes()} = {manual_beta:.4f}")
        
        # 線形グラフでも確認
        print(f"\n📏 線形グラフ (パス):")
        path = nx.path_graph(4)  # 1-2-3-4
        print(f"  ノード数: {path.number_of_nodes()}")
        print(f"  エッジ数: {path.number_of_edges()}")
        
        mu_path = ConnectivityMetrics.calculate_mu_index(path)
        beta_path = ConnectivityMetrics.calculate_beta_index(path)
        
        print(f"  μ指数: {mu_path}")
        print(f"  β指数: {beta_path:.4f}")
        
        manual_mu_path = path.number_of_edges() - path.number_of_nodes() + nx.number_connected_components(path)
        manual_beta_path = path.number_of_edges() / path.number_of_nodes()
        print(f"  手動 μ: {manual_mu_path}")
        print(f"  手動 β: {manual_beta_path:.4f}")
        
        print(f"\n✅ デバッグ完了")
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_connectivity_metrics()