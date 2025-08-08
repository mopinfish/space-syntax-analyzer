#!/usr/bin/env python3
"""
space-syntax-analyzer デモンストレーションスクリプト

このスクリプトは、拡張されたspace-syntax-analyzerの全機能を
デモンストレーションします。
"""

import logging
import os
from pathlib import Path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_basic_analysis():
    """基本分析のデモンストレーション"""
    print("="*50)
    print("🚀 基本分析デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer import SpaceSyntaxAnalyzer

        # アナライザーの初期化
        analyzer = SpaceSyntaxAnalyzer()
        
        # 渋谷駅周辺の分析
        print("\n📍 分析対象: 渋谷駅周辺")
        results = analyzer.analyze_place("Shibuya Station, Tokyo, Japan")
        
        # 結果の表示
        report = analyzer.generate_report(results, "渋谷駅周辺")
        print(report)
        
        # 可視化
        print("\n📊 可視化を表示中...")
        major_network, full_network = analyzer.get_network("Shibuya Station, Tokyo, Japan", "both")
        analyzer.visualize(major_network, full_network, results)
        
        # 結果のエクスポート
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        analyzer.export_results(results, str(output_dir / "basic_analysis.csv"))
        print(f"✅ 基本分析結果を {output_dir / 'basic_analysis.csv'} に保存")
        
    except Exception as e:
        logger.error(f"基本分析デモエラー: {e}")
        print(f"❌ 基本分析でエラーが発生しました: {e}")


def demo_enhanced_analysis():
    """拡張分析（軸線分析・可視領域分析）のデモンストレーション"""
    print("\n" + "="*50)
    print("🔬 拡張分析デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.enhanced_analyzer import (
            EnhancedSpaceSyntaxAnalyzer,
        )

        # 拡張アナライザーの初期化
        analyzer = EnhancedSpaceSyntaxAnalyzer(
            enable_axial_analysis=True,
            enable_visibility_analysis=True,
            visibility_radius=100.0
        )
        
        print("\n📍 包括的分析実行中: 京都駅周辺")
        print("   - 基本ネットワーク分析")
        print("   - 軸線分析（Axial Analysis）")
        print("   - 可視領域分析（Visibility Analysis）")
        
        # 包括的分析の実行
        results = analyzer.analyze_comprehensive("Kyoto Station, Kyoto, Japan")
        
        # 統合評価の表示
        evaluation = results.get('integrated_evaluation', {})
        print(f"\n🎯 統合評価結果:")
        print(f"   総合スコア: {evaluation.get('overall_score', 0):.1f}/100")
        print(f"   評価レベル: {evaluation.get('evaluation_level', 'N/A')}")
        print(f"   回遊性スコア: {evaluation.get('connectivity_score', 0):.1f}/100")
        print(f"   アクセス性スコア: {evaluation.get('accessibility_score', 0):.1f}/100")
        print(f"   効率性スコア: {evaluation.get('efficiency_score', 0):.1f}/100")
        
        # 軸線分析結果の表示
        axial_analysis = results.get('axial_analysis', {})
        if axial_analysis:
            network_metrics = axial_analysis.get('network_metrics', {})
            print(f"\n🔍 軸線分析結果:")
            print(f"   軸線数: {network_metrics.get('axial_lines', 0)}")
            print(f"   格子度: {network_metrics.get('grid_axiality', 0):.3f}")
            print(f"   循環度: {network_metrics.get('axial_ringiness', 0):.3f}")
        
        # 可視領域分析結果の表示
        visibility_analysis = results.get('visibility_analysis', {})
        if visibility_analysis:
            field_stats = visibility_analysis.get('visibility_field', {}).get('field_statistics', {})
            if field_stats:
                print(f"\n👁️ 可視領域分析結果:")
                print(f"   サンプリング点数: {field_stats.get('total_sampling_points', 0)}")
                print(f"   平均可視面積: {field_stats.get('mean_visible_area', 0):.1f}m²")
                print(f"   可視領域変動係数: {field_stats.get('std_visible_area', 0):.1f}")
        
        # 包括的可視化
        print("\n📊 包括的可視化を表示中...")
        output_dir = Path("demo_output")
        analyzer.visualize_comprehensive(
            results, 
            save_path=str(output_dir / "comprehensive_analysis.png")
        )
        
        # 学術レポートの生成
        print("\n📝 学術レポート生成中...")
        report_path = analyzer.generate_academic_report(
            results, 
            str(output_dir / "academic_report.txt"),
            include_visualizations=True
        )
        print(f"✅ 学術レポートを {report_path} に保存")
        
    except Exception as e:
        logger.error(f"拡張分析デモエラー: {e}")
        print(f"❌ 拡張分析でエラーが発生しました: {e}")


def demo_comparison_analysis():
    """複数地域比較分析のデモンストレーション"""
    print("\n" + "="*50)
    print("🏙️ 複数地域比較分析デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.enhanced_analyzer import (
            EnhancedSpaceSyntaxAnalyzer,
        )
        
        analyzer = EnhancedSpaceSyntaxAnalyzer(
            enable_axial_analysis=True,
            enable_visibility_analysis=False  # 高速化のため無効
        )
        
        # 比較対象地域
        locations = [
            "Shibuya Station, Tokyo, Japan",
            "Kyoto Station, Kyoto, Japan", 
            "Osaka Station, Osaka, Japan"
        ]
        location_names = ["渋谷", "京都", "大阪"]
        
        print(f"\n📍 比較分析対象: {', '.join(location_names)}")
        print("   各地域の包括的分析を実行中...")
        
        # 比較分析の実行
        comparison_results = analyzer.compare_locations(locations, location_names)
        
        # 比較結果の表示
        comparison_analysis = comparison_results.get('comparison_analysis', {})
        rankings = comparison_analysis.get('rankings', {})
        
        if rankings:
            print(f"\n🏆 各指標のランキング:")
            
            for metric, ranking in rankings.items():
                print(f"\n   {metric}:")
                for i, (location, value) in enumerate(ranking[:3], 1):
                    print(f"     {i}位: {location} ({value:.2f})")
        
        # 特徴的地域の表示
        characteristic_locations = comparison_analysis.get('characteristic_locations', {})
        if characteristic_locations:
            print(f"\n🌟 特徴的地域:")
            for characteristic, location in characteristic_locations.items():
                print(f"   {characteristic}: {location}")
        
        # 比較ダッシュボードの作成
        print("\n📊 比較ダッシュボード作成中...")
        output_dir = Path("demo_output")
        analyzer.create_comparison_dashboard(
            comparison_results,
            save_path=str(output_dir / "comparison_dashboard.png")
        )
        print(f"✅ 比較ダッシュボードを {output_dir / 'comparison_dashboard.png'} に保存")
        
    except Exception as e:
        logger.error(f"比較分析デモエラー: {e}")
        print(f"❌ 比較分析でエラーが発生しました: {e}")


def demo_axial_analysis_detailed():
    """軸線分析の詳細デモンストレーション"""
    print("\n" + "="*50)
    print("🔍 軸線分析詳細デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer import SpaceSyntaxAnalyzer
        from space_syntax_analyzer.core.axial import AxialAnalyzer
        from space_syntax_analyzer.core.enhanced_visualization import EnhancedVisualizer

        # 基本ネットワークの取得
        base_analyzer = SpaceSyntaxAnalyzer()
        major_network, _ = base_analyzer.get_network("Ginza, Tokyo, Japan", "major")
        
        print("\n📍 軸線分析対象: 銀座地区")
        print(f"   基本ネットワーク: {major_network.number_of_nodes()}ノード, {major_network.number_of_edges()}エッジ")
        
        # 軸線分析の実行
        axial_analyzer = AxialAnalyzer()
        
        print("\n🔧 軸線マップ作成中...")
        axial_map = axial_analyzer.create_axial_map(major_network)
        print(f"   軸線マップ: {axial_map.number_of_nodes()}軸線")
        
        print("\n📊 Integration Value計算中...")
        global_integration = axial_analyzer.analyze_global_integration(axial_map)
        local_integration = axial_analyzer.analyze_local_integration(axial_map, radius=3)
        
        print(f"   Global Integration: {len(global_integration)}軸線")
        print(f"   Local Integration (R3): {len(local_integration)}軸線")
        
        # 統計の表示
        if global_integration:
            values = list(global_integration.values())
            print(f"\n📈 Integration Value統計:")
            print(f"   平均: {sum(values)/len(values):.3f}")
            print(f"   最大: {max(values):.3f}")
            print(f"   最小: {min(values):.3f}")
        
        # 形態指標の計算
        network_metrics = axial_analyzer.calculate_axial_network_metrics(axial_map)
        print(f"\n🏗️ 軸線ネットワーク形態指標:")
        print(f"   格子度(GA): {network_metrics.get('grid_axiality', 0):.3f}")
        print(f"   循環度(AR): {network_metrics.get('axial_ringiness', 0):.3f}")
        print(f"   分節度(AA): {network_metrics.get('axial_articulation', 0):.3f}")
        
        # 可視化
        print("\n📊 軸線分析可視化中...")
        visualizer = EnhancedVisualizer()
        output_dir = Path("demo_output")
        
        # Integration Value分布の可視化
        visualizer.plot_integration_value_distribution(
            global_integration,
            title="銀座地区 Integration Value分布",
            save_path=str(output_dir / "integration_distribution.png")
        )
        
        # 軸線マップの可視化
        visualizer.plot_axial_map_with_integration(
            axial_map,
            global_integration,
            title="銀座地区 軸線マップ (Global Integration)",
            save_path=str(output_dir / "axial_map.png")
        )
        
        print(f"✅ 軸線分析可視化結果を {output_dir} に保存")
        
    except Exception as e:
        logger.error(f"軸線分析詳細デモエラー: {e}")
        print(f"❌ 軸線分析詳細デモでエラーが発生しました: {e}")


def demo_visibility_analysis_detailed():
    """可視領域分析の詳細デモンストレーション"""
    print("\n" + "="*50)
    print("👁️ 可視領域分析詳細デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer import SpaceSyntaxAnalyzer
        from space_syntax_analyzer.core.enhanced_visualization import EnhancedVisualizer
        from space_syntax_analyzer.core.visibility import VisibilityAnalyzer

        # 基本ネットワークの取得
        base_analyzer = SpaceSyntaxAnalyzer()
        major_network, _ = base_analyzer.get_network("Harajuku, Tokyo, Japan", "major")
        
        print("\n📍 可視領域分析対象: 原宿地区")
        print(f"   基本ネットワーク: {major_network.number_of_nodes()}ノード, {major_network.number_of_edges()}エッジ")
        
        # 可視領域分析の実行
        visibility_analyzer = VisibilityAnalyzer(visibility_radius=75.0)
        
        print("\n🔍 可視領域フィールド分析中...")
        visibility_field = visibility_analyzer.analyze_visibility_field(
            major_network, sampling_distance=30.0
        )
        
        field_stats = visibility_field.get('field_statistics', {})
        if field_stats:
            print(f"   サンプリング点数: {field_stats.get('total_sampling_points', 0)}")
            print(f"   平均可視面積: {field_stats.get('mean_visible_area', 0):.1f}m²")
            print(f"   可視面積範囲: {field_stats.get('min_visible_area', 0):.1f} - {field_stats.get('max_visible_area', 0):.1f}m²")
        
        # 変動性指標の表示
        variability_metrics = visibility_field.get('variability_metrics', {})
        if variability_metrics:
            print(f"\n📊 可視領域変動性:")
            print(f"   面積変動係数: {variability_metrics.get('area_coefficient_variation', 0):.3f}")
            print(f"   多様性指標: {variability_metrics.get('spatial_diversity_index', 0):.3f}")
        
        print("\n🔗 視覚的接続性分析中...")
        visual_connectivity = visibility_analyzer.analyze_visual_connectivity(major_network)
        
        network_metrics = visual_connectivity.get('network_metrics', {})
        if network_metrics:
            print(f"   視覚的ノード数: {network_metrics.get('visual_nodes', 0)}")
            print(f"   視覚的接続数: {network_metrics.get('visual_edges', 0)}")
            print(f"   平均視覚的接続性: {network_metrics.get('avg_visual_connectivity', 0):.3f}")
        
        # 単一点での詳細Isovist分析
        print("\n👁️ 代表点でのIsovist分析...")
        sampling_points = visibility_field.get('sampling_points', [])
        if sampling_points:
            center_point = sampling_points[len(sampling_points)//2]  # 中央付近の点
            isovist_result = visibility_analyzer.calculate_isovist(center_point, major_network)
            
            print(f"   観測点: ({center_point[0]:.1f}, {center_point[1]:.1f})")
            print(f"   可視面積: {isovist_result.get('visible_area', 0):.1f}m²")
            print(f"   コンパクト性: {isovist_result.get('compactness', 0):.3f}")
            print(f"   遮蔽性: {isovist_result.get('occlusivity', 0):.3f}")
        
        # 可視化
        print("\n📊 可視領域分析可視化中...")
        visualizer = EnhancedVisualizer()
        output_dir = Path("demo_output")
        
        visualizer.plot_visibility_field(
            visibility_field,
            title="原宿地区 可視領域フィールド分析",
            save_path=str(output_dir / "visibility_field.png")
        )
        
        # データエクスポート
        print("\n💾 可視領域データエクスポート中...")
        visibility_analyzer.export_visibility_results(
            visibility_field,
            str(output_dir / "visibility_data.csv"),
            format_type="csv"
        )
        
        print(f"✅ 可視領域分析結果を {output_dir} に保存")
        
    except Exception as e:
        logger.error(f"可視領域分析詳細デモエラー: {e}")
        print(f"❌ 可視領域分析詳細デモでエラーが発生しました: {e}")


def demo_performance_test():
    """パフォーマンステストのデモンストレーション"""
    print("\n" + "="*50)
    print("⚡ パフォーマンステストデモンストレーション")
    print("="*50)
    
    try:
        import time

        from space_syntax_analyzer.core.enhanced_analyzer import (
            EnhancedSpaceSyntaxAnalyzer,
        )

        # 異なるサイズの地域でテスト
        test_locations = [
            ("小規模", "Harajuku Station, Tokyo, Japan"),
            ("中規模", "Shibuya, Tokyo, Japan"), 
            ("大規模", "Tokyo Station, Tokyo, Japan"),
        ]
        
        analyzer = EnhancedSpaceSyntaxAnalyzer(
            enable_axial_analysis=True,
            enable_visibility_analysis=True
        )
        
        performance_results = []
        
        for scale, location in test_locations:
            print(f"\n🧪 {scale}地域テスト: {location}")
            
            start_time = time.time()
            
            try:
                # 基本ネットワーク取得時間
                network_start = time.time()
                major_network, full_network = analyzer.get_network(location, "both")
                network_time = time.time() - network_start
                
                network_size = major_network.number_of_nodes()
                print(f"   ネットワークサイズ: {network_size}ノード")
                print(f"   ネットワーク取得時間: {network_time:.2f}秒")
                
                # 基本分析時間
                basic_start = time.time()
                area_ha = analyzer.network_manager.calculate_area_ha(major_network)
                basic_results = analyzer.analyze(major_network, full_network, area_ha)
                basic_time = time.time() - basic_start
                print(f"   基本分析時間: {basic_time:.2f}秒")
                
                # 軸線分析時間（中規模以下のみ）
                axial_time = 0
                if network_size < 1000:  # 大規模では軸線分析をスキップ
                    axial_start = time.time()
                    axial_results = analyzer._perform_axial_analysis(major_network)
                    axial_time = time.time() - axial_start
                    print(f"   軸線分析時間: {axial_time:.2f}秒")
                else:
                    print(f"   軸線分析: スキップ（大規模のため）")
                
                total_time = time.time() - start_time
                print(f"   総処理時間: {total_time:.2f}秒")
                
                performance_results.append({
                    'scale': scale,
                    'location': location,
                    'network_size': network_size,
                    'network_time': network_time,
                    'basic_time': basic_time,
                    'axial_time': axial_time,
                    'total_time': total_time,
                })
                
                print(f"   ✅ {scale}地域テスト完了")
                
            except Exception as e:
                print(f"   ❌ {scale}地域テストでエラー: {e}")
                continue
        
        # パフォーマンス結果の表示
        print(f"\n📊 パフォーマンステスト結果サマリー:")
        print(f"{'規模':<10} {'ノード数':<10} {'取得時間':<10} {'基本分析':<10} {'軸線分析':<10} {'総時間':<10}")
        print("-" * 60)
        
        for result in performance_results:
            print(f"{result['scale']:<10} "
                  f"{result['network_size']:<10} "
                  f"{result['network_time']:<10.2f} "
                  f"{result['basic_time']:<10.2f} "
                  f"{result['axial_time']:<10.2f} "
                  f"{result['total_time']:<10.2f}")
        
        # 推奨事項の表示
        print(f"\n💡 パフォーマンス推奨事項:")
        print(f"   - 1000ノード未満: 全機能利用可能")
        print(f"   - 1000-3000ノード: 軸線分析は慎重に実行")
        print(f"   - 3000ノード以上: 基本分析のみ推奨")
        
    except Exception as e:
        logger.error(f"パフォーマンステストエラー: {e}")
        print(f"❌ パフォーマンステストでエラーが発生しました: {e}")


def main():
    """メインデモンストレーション関数"""
    print("🌟 space-syntax-analyzer 拡張機能デモンストレーション")
    print("="*70)
    
    # 出力ディレクトリの作成
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir.absolute()}")
    
    # デモメニュー
    demos = [
        ("1", "基本分析デモ", demo_basic_analysis),
        ("2", "拡張分析デモ（軸線・可視領域）", demo_enhanced_analysis),
        ("3", "複数地域比較分析デモ", demo_comparison_analysis),
        ("4", "軸線分析詳細デモ", demo_axial_analysis_detailed),
        ("5", "可視領域分析詳細デモ", demo_visibility_analysis_detailed),
        ("6", "パフォーマンステストデモ", demo_performance_test),
        ("a", "全デモ実行", None),
    ]
    
    print(f"\n📋 利用可能なデモ:")
    for code, name, _ in demos:
        print(f"   {code}: {name}")
    
    # 自動実行モード（環境変数で制御）
    auto_mode = os.getenv('DEMO_AUTO_MODE', 'false').lower() == 'true'
    
    if auto_mode:
        print(f"\n🤖 自動実行モード: 基本デモのみ実行")
        demo_basic_analysis()
    else:
        choice = input(f"\n選択してください (1-6, a, q=終了): ").strip().lower()
        
        if choice == 'q':
            print("👋 デモンストレーション終了")
            return
        elif choice == 'a':
            print(f"\n🚀 全デモを順次実行します...")
            for code, name, func in demos[:-1]:  # 'a'以外の全て
                if func:
                    print(f"\n▶️ {name} 開始")
                    func()
                    print(f"✅ {name} 完了")
        else:
            # 個別デモの実行
            for code, name, func in demos:
                if choice == code and func:
                    print(f"\n▶️ {name} 開始")
                    func()
                    print(f"✅ {name} 完了")
                    break
            else:
                print(f"❌ 無効な選択: {choice}")
    
    print(f"\n🎉 デモンストレーション完了!")
    print(f"📁 結果ファイルは {output_dir.absolute()} に保存されています")


if __name__ == "__main__":
    main()