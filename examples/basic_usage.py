"""
space-syntax-analyzer 最終修正版基本使用例

OSMnx v2.0完全対応版
"""

from space_syntax_analyzer import (
    SpaceSyntaxAnalyzer,
    calculate_bbox_area,
    check_osmnx_version,
    create_bbox_from_center,
    debug_network_info,
    estimate_processing_time,
    generate_comparison_summary,
    setup_logging,
)


def basic_analysis_example():
    """基本的な分析の例"""
    print("=== 基本的な分析例 ===")

    # ロギング設定
    setup_logging("INFO")

    # アナライザーの初期化
    analyzer = SpaceSyntaxAnalyzer()

    try:
        # 渋谷地域を分析
        print("渋谷地域を分析中...")
        results = analyzer.analyze_place("Shibuya, Tokyo, Japan")

        if results['metadata']['analysis_status'] == 'success':
            # レポート生成
            report = analyzer.generate_report(results, "渋谷駅周辺")
            print(report)

            # 結果をCSVで保存
            analyzer.export_results(results, "shibuya_analysis.csv")
            print("✅ 分析結果をshibuya_analysis.csvに保存しました")
        else:
            print(f"❌ 分析失敗: {results['metadata'].get('error_message', '不明なエラー')}")

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
            "Shinjuku, Tokyo, Japan", return_networks=True
        )

        if results['metadata']['analysis_status'] == 'success':
            # ネットワーク比較表示
            analyzer.visualize(
                major_net, full_net, results, save_path="shinjuku_networks.png"
            )

            print("✅ 可視化結果を保存しました:")
            print("- shinjuku_networks.png")
        else:
            print(f"❌ 分析失敗: {results['metadata'].get('error_message', '不明なエラー')}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


def custom_area_analysis():
    """カスタムエリア分析の例"""
    print("\n=== カスタムエリア分析例 ===")

    # カスタム設定でアナライザーを初期化
    analyzer = SpaceSyntaxAnalyzer(
        width_threshold=6.0,  # 6m以上を主要道路とする
        network_type="walk",  # 歩行者ネットワーク
    )

    try:
        # 東京駅から1km範囲の分析
        print("東京駅周辺1km範囲を分析中...")
        tokyo_station_coords = (35.6812, 139.7671)  # 東京駅の座標
        bbox = create_bbox_from_center(
            tokyo_station_coords[0], tokyo_station_coords[1], distance_km=1.0
        )

        results = analyzer.analyze_place(bbox)

        if results['metadata']['analysis_status'] == 'success':
            # 分析サマリーの生成
            if results.get('major_network') and results.get('full_network'):
                summary = generate_comparison_summary(
                    results['major_network'], results['full_network']
                )

                print("\n分析サマリー:")
                for key, value in summary.items():
                    print(f"- {key}: {value}")

            # Excel形式で保存
            analyzer.export_results(
                results, "tokyo_station_analysis.xlsx", format_type="excel"
            )
            print("✅ 結果をtokyo_station_analysis.xlsxに保存しました")
        else:
            print(f"❌ 分析失敗: {results['metadata'].get('error_message', '不明なエラー')}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


def batch_analysis_example():
    """複数地域の一括分析例"""
    print("\n=== 複数地域一括分析例 ===")

    analyzer = SpaceSyntaxAnalyzer()

    # 分析対象地域（小さめの範囲に限定してエラーを避ける）
    locations = [
        "Shibuya Station, Tokyo, Japan",
        "Shinjuku Station, Tokyo, Japan", 
    ]

    all_results = {}

    for location in locations:
        try:
            print(f"{location}を分析中...")
            results = analyzer.analyze_place(location)
            
            if results['metadata']['analysis_status'] == 'success':
                all_results[location] = results
                print(f"✅ {location}の分析完了")
            else:
                print(f"❌ {location}の分析失敗: {results['metadata'].get('error_message', '不明なエラー')}")

        except Exception as e:
            print(f"❌ {location}の分析でエラー: {e}")

    # 比較レポートの生成
    print("\n=== 比較レポート ===")
    for location, results in all_results.items():
        print(f"\n{location}:")
        if results.get('major_network'):
            metrics = results['major_network']
            print(f"  α指数: {metrics.get('alpha_index', 0):.1f}%")
            print(f"  β指数: {metrics.get('beta_index', 0):.2f}")
            print(f"  平均迂回率: {metrics.get('avg_circuity', 0):.2f}")
            print(f"  道路密度: {metrics.get('road_density', 0):.1f}")


def network_export_example():
    """ネットワークエクスポートの例"""
    print("\n=== ネットワークエクスポート例 ===")

    analyzer = SpaceSyntaxAnalyzer()

    try:
        # 原宿駅周辺のネットワーク取得
        print("原宿駅周辺のネットワークを取得中...")
        major_net, full_net = analyzer.get_network(
            "Harajuku Station, Tokyo, Japan", "both"
        )

        if major_net is not None or full_net is not None:
            # 各種形式でエクスポート
            if major_net:
                analyzer.network_manager.export_network(
                    major_net, "harajuku_major_roads.geojson", "geojson"
                )
                analyzer.network_manager.export_network(
                    major_net, "harajuku_major_roads.graphml", "graphml"
                )

            if full_net:
                analyzer.network_manager.export_network(
                    full_net, "harajuku_all_roads.geojson", "geojson"
                )

            print("✅ ネットワークをエクスポートしました:")
            if major_net:
                print("- harajuku_major_roads.geojson")
                print("- harajuku_major_roads.graphml")
            if full_net:
                print("- harajuku_all_roads.geojson")
        else:
            print("❌ ネットワーク取得に失敗しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


def comprehensive_analysis_example():
    """包括的な分析例"""
    print("\n=== 包括的な分析例 ===")
    
    # 異なる設定で複数の分析を実行
    configs = [
        {"network_type": "drive", "width_threshold": 6.0, "name": "車両用道路"},
        {"network_type": "walk", "width_threshold": 3.0, "name": "歩行者用道路"},
    ]
    
    location = "Shibuya Station, Tokyo, Japan"
    
    for config in configs:
        try:
            print(f"\n{config['name']}ネットワークを分析中...")
            
            analyzer = SpaceSyntaxAnalyzer(
                network_type=config["network_type"],
                width_threshold=config["width_threshold"]
            )
            
            results = analyzer.analyze_place(location)
            
            if results['metadata']['analysis_status'] == 'success':
                major = results.get('major_network', {})
                full = results.get('full_network', {})
                
                print(f"  {config['name']} 分析結果:")
                print(f"    主要道路: {major.get('node_count', 0)} ノード")
                print(f"    全道路: {full.get('node_count', 0)} ノード")
                print(f"    α指数: {major.get('alpha_index', 0):.1f}%")
                print(f"    連結性: {major.get('connectivity_ratio', 0):.2f}")
            else:
                print(f"  ❌ {config['name']}ネットワーク分析失敗")
                
        except Exception as e:
            print(f"  ❌ {config['name']}ネットワーク分析エラー: {e}")


def debug_and_diagnostic_example():
    """デバッグと診断の例"""
    print("\n=== デバッグと診断例 ===")

    # バージョン情報の確認
    print("システム情報:")
    version_info = check_osmnx_version()
    for key, value in version_info.items():
        print(f"  {key}: {value}")
    
    # 小さなエリアでテスト
    analyzer = SpaceSyntaxAnalyzer()
    
    try:
        # 東京駅周辺の小さなエリア
        tokyo_coords = (35.6812, 139.7671)
        bbox = create_bbox_from_center(tokyo_coords[0], tokyo_coords[1], 0.2)  # 200m範囲
        
        print(f"\n小範囲テスト (bbox: {bbox}):")
        
        results, (major_net, full_net) = analyzer.analyze_place(bbox, return_networks=True)
        
        # ネットワークのデバッグ情報
        debug_network_info(major_net, "主要道路")
        debug_network_info(full_net, "全道路")
        
        if results['metadata']['analysis_status'] == 'success':
            print("✅ 小範囲テスト成功")
        else:
            print(f"❌ 小範囲テスト失敗: {results['metadata'].get('error_message')}")
            
    except Exception as e:
        print(f"❌ デバッグテストエラー: {e}")


def performance_test_example():
    """パフォーマンステストの例"""
    print("\n=== パフォーマンステスト例 ===")
    
    import time

    # 異なるサイズのエリアでテスト
    test_areas = [
        {"name": "小エリア", "coords": (35.6812, 139.7671), "distance": 0.2},
        {"name": "中エリア", "coords": (35.6580, 139.7016), "distance": 0.5},
    ]
    
    analyzer = SpaceSyntaxAnalyzer()
    
    for area in test_areas:
        try:
            bbox = create_bbox_from_center(
                area["coords"][0], area["coords"][1], area["distance"]
            )
            
            area_km2 = calculate_bbox_area(bbox)
            estimated_time = estimate_processing_time(bbox)
            
            print(f"\n{area['name']} ({area_km2:.2f} km²):")
            print(f"  推定処理時間: {estimated_time}")
            
            start_time = time.time()
            results = analyzer.analyze_place(bbox)
            actual_time = time.time() - start_time
            
            if results['metadata']['analysis_status'] == 'success':
                print(f"  実際の処理時間: {actual_time:.1f}秒")
                major = results.get('major_network', {})
                print(f"  取得ノード数: {major.get('node_count', 0)}")
            else:
                print(f"  分析失敗: {results['metadata'].get('error_message')}")
                
        except Exception as e:
            print(f"  エラー: {e}")


if __name__ == "__main__":
    """実行例"""
    print("space-syntax-analyzer 最終修正版使用例を実行します")
    print("=" * 70)

    # 各例を順次実行（エラーが起きても続行）
    try:
        basic_analysis_example()
    except Exception as e:
        print(f"基本分析例でエラー: {e}")

    try:
        visualization_example() 
    except Exception as e:
        print(f"可視化例でエラー: {e}")

    try:
        custom_area_analysis()
    except Exception as e:
        print(f"カスタムエリア分析例でエラー: {e}")

    try:
        batch_analysis_example()
    except Exception as e:
        print(f"バッチ分析例でエラー: {e}")

    try:
        network_export_example()
    except Exception as e:
        print(f"ネットワークエクスポート例でエラー: {e}")

    try:
        comprehensive_analysis_example()
    except Exception as e:
        print(f"包括的分析例でエラー: {e}")

    try:
        debug_and_diagnostic_example()
    except Exception as e:
        print(f"デバッグ診断例でエラー: {e}")

    try:
        performance_test_example()
    except Exception as e:
        print(f"パフォーマンステスト例でエラー: {e}")

    print("\n" + "=" * 70)
    print("すべての例の実行が完了しました")
    
    print("\n🎉 Space Syntax Analyzer の機能紹介:")
    print("  ✅ OSMnx v2.0完全対応")
    print("  ✅ 複数ネットワークタイプサポート")  
    print("  ✅ 包括的なSpace Syntax指標")
    print("  ✅ 高度な可視化機能")
    print("  ✅ 複数出力フォーマット")
    print("  ✅ バッチ処理機能")
    print("  ✅ エラーハンドリングと診断")
    print("\n詳細な使用方法については、ドキュメントを参照してください。")