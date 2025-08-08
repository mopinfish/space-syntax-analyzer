"""
space-syntax-analyzer 基本使用例

このスクリプトは、space-syntax-analyzerの基本的な使用方法を示します。
"""

from space_syntax_analyzer import SpaceSyntaxAnalyzer
from space_syntax_analyzer.utils import setup_logging, create_bbox_from_center


def basic_analysis_example():
    """基本的な分析の例"""
    print("=== 基本的な分析例 ===")
    
    # ロギング設定
    setup_logging("INFO")
    
    # アナライザーの初期化
    analyzer = SpaceSyntaxAnalyzer()
    
    try:
        # 渋谷駅周辺の分析
        print("渋谷駅周辺を分析中...")
        results = analyzer.analyze_place("Shibuya Station, Tokyo, Japan")
        
        # レポート生成
        report = analyzer.generate_report(results, "渋谷駅周辺")
        print(report)
        
        # 結果をCSVで保存
        analyzer.export_results(results, "shibuya_analysis.csv")
        print("分析結果をshibuya_analysis.csvに保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def visualization_example():
    """可視化の例"""
    print("\n=== 可視化例 ===")
    
    analyzer = SpaceSyntaxAnalyzer()
    
    try:
        # ネットワークと結果を取得
        print("新宿駅周辺を分析・可視化中...")
        results, (major_net, full_net) = analyzer.analyze_place(
            "Shinjuku Station, Tokyo, Japan",
            return_networks=True
        )
        
        # ネットワーク比較表示
        analyzer.visualize(major_net, full_net, results, save_path="shinjuku_networks.png")
        
        # 指標比較チャート
        analyzer.visualizer.plot_metrics_comparison(results, save_path="shinjuku_metrics.png")
        
        print("可視化結果を保存しました:")
        print("- shinjuku_networks.png")
        print("- shinjuku_metrics.png")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def custom_area_analysis():
    """カスタムエリア分析の例"""
    print("\n=== カスタムエリア分析例 ===")
    
    # カスタム設定でアナライザーを初期化
    analyzer = SpaceSyntaxAnalyzer(
        width_threshold=6.0,  # 6m以上を主要道路とする
        network_type="walk"   # 歩行者ネットワーク
    )
    
    try:
        # 東京駅から1km範囲の分析
        print("東京駅周辺1km範囲を分析中...")
        tokyo_station_coords = (35.6812, 139.7671)  # 東京駅の座標
        bbox = create_bbox_from_center(
            tokyo_station_coords[0], 
            tokyo_station_coords[1], 
            distance_km=1.0
        )
        
        results = analyzer.analyze_place(bbox)
        
        # 分析サマリーの生成
        if "major_network" in results and "full_network" in results:
            from space_syntax_analyzer.utils import generate_comparison_summary
            summary = generate_comparison_summary(
                results["major_network"],
                results["full_network"]
            )
            
            print("\n分析サマリー:")
            for key, value in summary.items():
                print(f"- {key}: {value}")
                
        # Excel形式で保存
        analyzer.export_results(results, "tokyo_station_analysis.xlsx", format_type="excel")
        print("結果をtokyo_station_analysis.xlsxに保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def batch_analysis_example():
    """複数地域の一括分析例"""
    print("\n=== 複数地域一括分析例 ===")
    
    analyzer = SpaceSyntaxAnalyzer()
    
    # 分析対象地域
    locations = [
        "Shibuya, Tokyo, Japan",
        "Shinjuku, Tokyo, Japan",
        "Harajuku, Tokyo, Japan",
        "Akihabara, Tokyo, Japan"
    ]
    
    all_results = {}
    
    for location in locations:
        try:
            print(f"{location}を分析中...")
            results = analyzer.analyze_place(location)
            all_results[location] = results
            print(f"{location}の分析完了")
            
        except Exception as e:
            print(f"{location}の分析でエラー: {e}")
            
    # 比較レポートの生成
    print("\n=== 比較レポート ===")
    for location, results in all_results.items():
        print(f"\n{location}:")
        if "major_network" in results:
            metrics = results["major_network"]
            print(f"  α指数: {metrics.get('alpha_index', 0):.1f}%")
            print(f"  β指数: {metrics.get('beta_index', 0):.2f}")
            print(f"  迂回率: {metrics.get('avg_circuity', 0):.2f}")
            print(f"  道路密度: {metrics.get('road_density', 0):.1f}m/ha")


def network_export_example():
    """ネットワークエクスポートの例"""
    print("\n=== ネットワークエクスポート例 ===")
    
    analyzer = SpaceSyntaxAnalyzer()
    
    try:
        # 原宿駅周辺のネットワーク取得
        print("原宿駅周辺のネットワークを取得中...")
        major_net, full_net = analyzer.get_network("Harajuku Station, Tokyo, Japan", "both")
        
        # 各種形式でエクスポート
        analyzer.network_manager.export_network(
            major_net, "harajuku_major_roads.geojson", "geojson"
        )
        analyzer.network_manager.export_network(
            full_net, "harajuku_all_roads.geojson", "geojson"
        )
        analyzer.network_manager.export_network(
            major_net, "harajuku_major_roads.graphml", "graphml"
        )
        
        print("ネットワークをエクスポートしました:")
        print("- harajuku_major_roads.geojson")
        print("- harajuku_all_roads.geojson")
        print("- harajuku_major_roads.graphml")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    """実行例"""
    print("space-syntax-analyzer 使用例を実行します")
    print("=" * 50)
    
    # 各例を順次実行
    basic_analysis_example()
    visualization_example() 
    custom_area_analysis()
    batch_analysis_example()
    network_export_example()
    
    print("\n" + "=" * 50)
    print("すべての例の実行が完了しました")