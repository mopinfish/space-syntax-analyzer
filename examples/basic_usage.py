"""
space-syntax-analyzer 修正版基本使用例

このスクリプトは、space-syntax-analyzerの基本的な使用方法を示します。
エラー修正を含んだ版です。
"""

from space_syntax_analyzer import SpaceSyntaxAnalyzer
from space_syntax_analyzer.utils import create_bbox_from_center, setup_logging
import osmnx as ox


def basic_analysis_example():
    """基本的な分析の例（修正版）"""
    print("=== 基本的な分析例（修正版） ===")

    # ロギング設定
    setup_logging("INFO")

    # アナライザーの初期化
    analyzer = SpaceSyntaxAnalyzer()

    try:
        # より広域の地名で検索（駅名ではなく地域名を使用）
        print("渋谷地域を分析中...")
        results = analyzer.analyze_place("Shibuya, Tokyo, Japan")

        # レポート生成
        report = analyzer.generate_report(results, "渋谷地域")
        print(report)

        # 結果をCSVで保存
        analyzer.export_results(results, "shibuya_analysis.csv")
        print("分析結果をshibuya_analysis.csvに保存しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        # 代替手段として座標を使った分析
        try:
            print("代替手段として座標を使用して分析中...")
            # 渋谷駅の座標
            shibuya_coords = (35.6580, 139.7016)
            bbox = create_bbox_from_center(
                shibuya_coords[0], shibuya_coords[1], distance_km=0.5
            )
            results = analyzer.analyze_place(bbox)
            
            if results:
                # レポート生成
                report = analyzer.generate_report(results, "渋谷駅周辺（座標指定）")
                print(report)
                
                # 結果をCSVで保存
                analyzer.export_results(results, "shibuya_coords_analysis.csv")
                print("分析結果をshibuya_coords_analysis.csvに保存しました")
            
        except Exception as alt_e:
            print(f"代替手段でもエラーが発生しました: {alt_e}")


def visualization_example():
    """可視化の例（修正版）"""
    print("\n=== 可視化例（修正版） ===")

    analyzer = SpaceSyntaxAnalyzer()

    try:
        # 地域名で検索（駅名ではなく）
        print("新宿地域を分析・可視化中...")
        results, (major_net, full_net) = analyzer.analyze_place(
            "Shinjuku, Tokyo, Japan", return_networks=True
        )

        # ネットワーク比較表示
        analyzer.visualize(
            major_net, full_net, results, save_path="shinjuku_networks.png"
        )

        # 指標比較チャート
        analyzer.visualizer.plot_metrics_comparison(
            results, save_path="shinjuku_metrics.png"
        )

        print("可視化結果を保存しました:")
        print("- shinjuku_networks.png")
        print("- shinjuku_metrics.png")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        # 座標を使った代替手段
        try:
            print("座標を使って代替分析中...")
            shinjuku_coords = (35.6896, 139.6917)
            bbox = create_bbox_from_center(
                shinjuku_coords[0], shinjuku_coords[1], distance_km=0.5
            )
            
            results, (major_net, full_net) = analyzer.analyze_place(
                bbox, return_networks=True
            )

            if results and major_net is not None and full_net is not None:
                # ネットワーク比較表示
                analyzer.visualize(
                    major_net, full_net, results, save_path="shinjuku_networks_coords.png"
                )

                print("代替可視化結果を保存しました:")
                print("- shinjuku_networks_coords.png")
            
        except Exception as alt_e:
            print(f"代替手段でもエラーが発生しました: {alt_e}")


def custom_area_analysis():
    """カスタムエリア分析の例（修正版）"""
    print("\n=== カスタムエリア分析例（修正版） ===")

    # カスタム設定でアナライザーを初期化
    analyzer = SpaceSyntaxAnalyzer(
        width_threshold=6.0,  # 6m以上を主要道路とする
        network_type="walk",  # 歩行者ネットワーク
    )

    try:
        # 東京駅から1km範囲の分析（座標を直接使用）
        print("東京駅周辺1km範囲を分析中...")
        tokyo_station_coords = (35.6812, 139.7671)  # 東京駅の座標
        
        # 修正されたbbox作成方法
        bbox = create_bbox_from_center(
            tokyo_station_coords[0], tokyo_station_coords[1], distance_km=1.0
        )

        results = analyzer.analyze_place(bbox)

        # 分析サマリーの生成
        if results and "major_network" in results and "full_network" in results:
            from space_syntax_analyzer.utils import generate_comparison_summary

            summary = generate_comparison_summary(
                results["major_network"], results["full_network"]
            )

            print("\n分析サマリー:")
            for key, value in summary.items():
                print(f"- {key}: {value}")

        # Excel形式で保存
        analyzer.export_results(
            results, "tokyo_station_analysis.xlsx", format_type="excel"
        )
        print("結果をtokyo_station_analysis.xlsxに保存しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("詳細:", str(e))


def batch_analysis_example():
    """複数地域の一括分析例（修正版）"""
    print("\n=== 複数地域一括分析例（修正版） ===")

    # より許容度の高い設定でアナライザーを初期化
    analyzer = SpaceSyntaxAnalyzer(network_type="drive")

    # 分析対象地域（座標版も準備）
    locations = [
        ("Shibuya, Tokyo, Japan", (35.6580, 139.7016)),
        ("Shinjuku, Tokyo, Japan", (35.6896, 139.6917)),
        ("Harajuku, Tokyo, Japan", (35.6702, 139.7026)),
        ("Akihabara, Tokyo, Japan", (35.7022, 139.7742)),
    ]

    all_results = {}

    for location_name, coords in locations:
        try:
            print(f"{location_name}を分析中...")
            
            # まず地域名で試行
            try:
                results = analyzer.analyze_place(location_name)
                all_results[location_name] = results
                print(f"{location_name}の分析完了")
                continue
            except:
                print(f"{location_name}の地域名検索失敗、座標を使用中...")
            
            # 地域名がダメなら座標で試行
            bbox = create_bbox_from_center(coords[0], coords[1], distance_km=0.5)
            results = analyzer.analyze_place(bbox)
            all_results[location_name + " (座標)"] = results
            print(f"{location_name}の分析完了（座標使用）")

        except Exception as e:
            print(f"{location_name}の分析でエラー: {e}")

    # 比較レポートの生成
    print("\n=== 比較レポート ===")
    for location, results in all_results.items():
        print(f"\n{location}:")
        if results and "major_network" in results and results["major_network"]:
            metrics = results["major_network"]
            alpha_idx = metrics.get('alpha_index', 0)
            beta_idx = metrics.get('beta_index', 0)
            circuity = metrics.get('avg_circuity', 0)
            density = metrics.get('road_density', 0)
            
            print(f"  α指数: {alpha_idx:.1f}%")
            print(f"  β指数: {beta_idx:.2f}")
            print(f"  迂回率: {circuity:.2f}")
            print(f"  道路密度: {density:.1f}m/ha")
        else:
            print("  分析結果が不完全です")


def network_export_example():
    """ネットワークエクスポートの例（修正版）"""
    print("\n=== ネットワークエクスポート例（修正版） ===")

    analyzer = SpaceSyntaxAnalyzer()

    try:
        # 原宿地域のネットワーク取得（地域名使用）
        print("原宿地域のネットワークを取得中...")
        
        try:
            major_net, full_net = analyzer.get_network(
                "Harajuku, Tokyo, Japan", "both"
            )
        except:
            print("地域名検索失敗、座標を使用中...")
            # 座標での代替手段
            harajuku_coords = (35.6702, 139.7026)
            bbox = create_bbox_from_center(
                harajuku_coords[0], harajuku_coords[1], distance_km=0.3
            )
            major_net, full_net = analyzer.get_network(bbox, "both")

        # 各種形式でエクスポート
        if major_net is not None:
            analyzer.network_manager.export_network(
                major_net, "harajuku_major_roads.geojson", "geojson"
            )
            analyzer.network_manager.export_network(
                major_net, "harajuku_major_roads.graphml", "graphml"
            )
        
        if full_net is not None:
            analyzer.network_manager.export_network(
                full_net, "harajuku_all_roads.geojson", "geojson"
            )

        print("ネットワークをエクスポートしました:")
        if major_net is not None:
            print("- harajuku_major_roads.geojson")
            print("- harajuku_major_roads.graphml")
        if full_net is not None:
            print("- harajuku_all_roads.geojson")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


def safe_network_analysis_example():
    """安全なネットワーク分析例"""
    print("\n=== 安全なネットワーク分析例 ===")
    
    analyzer = SpaceSyntaxAnalyzer()
    
    # 明らかに動作するはずの座標での分析
    test_coords = [(35.6812, 139.7671)]  # 東京駅
    
    for i, coords in enumerate(test_coords):
        try:
            print(f"テスト地点 {i+1} ({coords}) を分析中...")
            bbox = create_bbox_from_center(coords[0], coords[1], distance_km=0.3)
            
            # OSMnxで直接ネットワーク取得してテスト
            try:
                # OSMnxで直接テスト
                test_net = ox.graph_from_bbox(
                    bbox, 
                    network_type="drive",
                    simplify=True,
                    retain_all=False
                )
                print(f"  OSMnx直接取得成功: {len(test_net.nodes)} ノード, {len(test_net.edges)} エッジ")
                
                # space-syntax-analyzerで分析
                results = analyzer.analyze_place(bbox)
                if results:
                    print(f"  space-syntax-analyzer分析成功")
                    # 基本統計の表示
                    if "major_network" in results and results["major_network"]:
                        metrics = results["major_network"]
                        print(f"    ネットワーク指標数: {len(metrics)}")
                else:
                    print(f"  space-syntax-analyzer分析失敗")
                    
            except Exception as inner_e:
                print(f"  内部エラー: {inner_e}")
                
        except Exception as e:
            print(f"テスト地点 {i+1} でエラー: {e}")


if __name__ == "__main__":
    """実行例"""
    print("space-syntax-analyzer 修正版使用例を実行します")
    print("=" * 60)

    # 各例を順次実行
    safe_network_analysis_example()  # 安全テストから開始
    basic_analysis_example()
    visualization_example()
    custom_area_analysis()
    batch_analysis_example()
    network_export_example()

    print("\n" + "=" * 60)
    print("すべての例の実行が完了しました")