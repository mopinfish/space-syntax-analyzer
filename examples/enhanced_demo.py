# enhanced_demo_final.py
"""
最終修正版拡張 Space Syntax Analyzer デモスクリプト

基底クラスの実際のシグネチャに合わせて修正し、高度な分析機能を統合
"""

import logging
import os
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_enhanced_comprehensive_analysis():
    """拡張版包括的分析のデモンストレーション"""
    print("="*60)
    print("🚀 拡張版 Space Syntax 包括的分析デモンストレーション")
    print("="*60)
    
    try:
        # まず基本アナライザーをテスト
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        # 拡張アナライザーの利用可能性をテスト
        enhanced_available = False
        try:
            # カスタムEnhancedSpaceSyntaxAnalyzerクラスを作成
            analyzer = create_enhanced_analyzer()
            enhanced_available = True
            print("✅ 拡張アナライザー初期化成功")
            analysis_type = "enhanced"
        except Exception as e:
            print(f"⚠️ 拡張アナライザー初期化エラー: {e}")
            print("🔄 基本アナライザーを使用")
            analyzer = SpaceSyntaxAnalyzer()
            analysis_type = "basic"
        
        # 複数の地名で試行
        test_locations = [
            "渋谷, 東京",
            "新宿, 東京", 
            "銀座, 東京",
            "東京駅, 東京"
        ]
        
        successful_analysis = None
        for location in test_locations:
            print(f"\n📍 分析試行: {location}")
            
            try:
                start_time = time.time()
                
                if enhanced_available:
                    # 拡張分析を実行
                    results = perform_enhanced_analysis(analyzer, location)
                else:
                    # 基本分析を実行
                    results = analyzer.analyze_place(location)
                
                end_time = time.time()
                
                # エラーチェック
                if results.get('error', False):
                    print(f"❌ 分析エラー: {results.get('error_message', '不明')}")
                    continue
                
                # 成功
                successful_analysis = (location, results, analysis_type)
                print(f"✅ 分析成功! (実行時間: {end_time - start_time:.1f}秒)")
                break
                
            except Exception as e:
                print(f"❌ 例外発生: {e}")
                logger.error(f"分析例外 ({location}): {e}")
                continue
        
        if successful_analysis:
            location, results, analysis_type = successful_analysis
            print(f"\n🎯 分析結果詳細: {location} ({analysis_type}分析)")
            
            # 分析結果の表示
            if analysis_type == "enhanced":
                display_enhanced_results(results)
            else:
                display_basic_results(results)
            
            # 結果のエクスポート
            export_results(analyzer, results, location, analysis_type)
            
            # 可視化（可能な場合）
            try_visualization(analyzer, results, location, analysis_type)
            
        else:
            print("❌ すべての地名で分析に失敗しました")
            print("💡 軸線分析単体デモを実行します")
            demo_axial_analysis_only()
            
    except ImportError as e:
        print(f"❌ モジュールインポートエラー: {e}")
        print("💡 基本機能のみでデモを実行します")
        demo_basic_analysis_fallback()
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        logger.error(f"拡張デモ実行エラー: {e}")


def create_enhanced_analyzer():
    """カスタム拡張アナライザーを作成"""
    from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
    from space_syntax_analyzer.core.axial import AxialAnalyzer
    from space_syntax_analyzer.core.visibility import VisibilityAnalyzer
    
    # 基本アナライザーを拡張
    class CustomEnhancedAnalyzer(SpaceSyntaxAnalyzer):
        def __init__(self):
            # 基底クラスの正しいシグネチャで初期化
            super().__init__(network_type="drive", width_threshold=4.0)
            
            # 拡張機能を追加
            self.enable_axial_analysis = True
            self.enable_visibility_analysis = True
            self.visibility_radius = 100.0
            
            # 拡張アナライザーの初期化
            self.axial_analyzer = AxialAnalyzer()
            self.visibility_analyzer = VisibilityAnalyzer(visibility_radius=100.0)
        
        def analyze_comprehensive(self, location, return_networks=False, analysis_level="global"):
            """包括的分析の実行"""
            try:
                logger.info(f"包括的分析開始: {location}")
                
                # 基本分析を実行
                basic_results = self.analyze_place(location, return_networks=True)
                
                if isinstance(basic_results, tuple):
                    results, networks = basic_results
                    major_network, full_network = networks
                else:
                    results = basic_results
                    major_network, full_network = None, None
                
                if results.get('error', False):
                    return results
                
                # 包括的分析結果の構築
                comprehensive_results = {
                    "location": str(location),
                    "basic_analysis": results,
                    "area_ha": results.get("area_ha", 0)
                }
                
                # 軸線分析
                if self.enable_axial_analysis and major_network:
                    logger.info("軸線分析実行中...")
                    axial_results = self._perform_axial_analysis(major_network, analysis_level)
                    comprehensive_results["axial_analysis"] = axial_results
                
                # 可視領域分析
                if self.enable_visibility_analysis and major_network:
                    logger.info("可視領域分析実行中...")
                    visibility_results = self._perform_visibility_analysis(major_network)
                    comprehensive_results["visibility_analysis"] = visibility_results
                
                # 統合評価
                comprehensive_results["integrated_evaluation"] = self._generate_integrated_evaluation(
                    comprehensive_results
                )
                
                logger.info("包括的分析完了")
                
                if return_networks:
                    return comprehensive_results, {"major_network": major_network, "full_network": full_network}
                else:
                    return comprehensive_results
                    
            except Exception as e:
                logger.error(f"包括的分析エラー: {e}")
                return {"error": True, "error_message": str(e)}
        
        def _perform_axial_analysis(self, network, analysis_level="global"):
            """軸線分析を実行"""
            try:
                # 軸線分析の実行
                axial_results = self.axial_analyzer.calculate_axial_summary(network)
                
                # 分析レベルに応じた追加計算
                axial_map = axial_results.get("axial_map")
                
                if axial_map and analysis_level in ["global", "both"]:
                    global_integration = self.axial_analyzer.analyze_global_integration(axial_map)
                    axial_results["global_integration"] = global_integration
                
                if axial_map and analysis_level in ["local", "both"]:
                    local_integration = self.axial_analyzer.analyze_local_integration(axial_map)
                    axial_results["local_integration"] = local_integration
                
                return axial_results
                
            except Exception as e:
                logger.warning(f"軸線分析実行エラー: {e}")
                return {"error": str(e)}
        
        def _perform_visibility_analysis(self, network):
            """可視領域分析を実行"""
            try:
                # 可視領域フィールド分析
                visibility_field = self.visibility_analyzer.analyze_visibility_field(
                    network, sampling_distance=50.0  # サンプリング間隔を大きくして処理軽減
                )
                
                # 視覚的接続性分析
                visual_connectivity = self.visibility_analyzer.analyze_visual_connectivity(network)
                
                return {
                    "visibility_field": visibility_field,
                    "visual_connectivity": visual_connectivity,
                }
                
            except Exception as e:
                logger.warning(f"可視領域分析実行エラー: {e}")
                return {"error": str(e)}
        
        def _generate_integrated_evaluation(self, results):
            """統合評価を生成"""
            try:
                basic_analysis = results.get("basic_analysis", {})
                
                # 主要道路ネットワークの指標を取得
                major_network = basic_analysis.get("major_network")
                if not major_network:
                    return {"error": "主要道路ネットワークデータがありません"}
                
                # 回遊性スコア
                alpha = major_network.get("alpha_index", 0)
                gamma = major_network.get("gamma_index", 0)
                connectivity_score = min((alpha + gamma) / 2, 100)
                
                # アクセス性スコア
                road_density = major_network.get("road_density", 0)
                intersection_density = major_network.get("intersection_density", 0)
                density_score = min((road_density / 10 + intersection_density * 5), 100)
                
                # 効率性スコア
                circuity = major_network.get("avg_circuity", 1.0)
                efficiency_score = max(0, min((2.0 - circuity) / 1.0 * 100, 100))
                
                # 総合スコア
                overall_score = (connectivity_score + density_score + efficiency_score) / 3
                
                # 評価レベルの判定
                if overall_score >= 80:
                    evaluation_level = "A - 優秀"
                elif overall_score >= 65:
                    evaluation_level = "B - 良好"
                elif overall_score >= 50:
                    evaluation_level = "C - 普通"
                elif overall_score >= 35:
                    evaluation_level = "D - 要改善"
                else:
                    evaluation_level = "E - 大幅改善必要"
                
                return {
                    "connectivity_score": connectivity_score,
                    "accessibility_score": density_score,
                    "efficiency_score": efficiency_score,
                    "overall_score": overall_score,
                    "evaluation_level": evaluation_level,
                    "analysis_timestamp": time.time()
                }
                
            except Exception as e:
                logger.warning(f"統合評価生成エラー: {e}")
                return {"error": str(e)}
    
    return CustomEnhancedAnalyzer()


def perform_enhanced_analysis(analyzer, location):
    """拡張分析を実行"""
    try:
        if hasattr(analyzer, 'analyze_comprehensive'):
            return analyzer.analyze_comprehensive(location, analysis_level="global")
        else:
            # フォールバック: 基本分析に軸線分析を追加
            basic_results = analyzer.analyze_place(location)
            
            if basic_results.get('error', False):
                return basic_results
            
            # 軸線分析を追加試行
            try:
                major_network, _ = analyzer.get_network(location, "major")
                if major_network and hasattr(analyzer, 'axial_analyzer'):
                    axial_results = analyzer.axial_analyzer.calculate_axial_summary(major_network)
                    basic_results["axial_analysis"] = axial_results
            except Exception as e:
                logger.warning(f"軸線分析追加エラー: {e}")
            
            return basic_results
            
    except Exception as e:
        logger.error(f"拡張分析実行エラー: {e}")
        return {"error": True, "error_message": str(e)}


def display_enhanced_results(results):
    """拡張分析結果の表示"""
    print(f"\n📊 拡張分析結果:")
    
    # 基本分析結果
    basic_analysis = results.get('basic_analysis', {})
    if basic_analysis:
        major_network = basic_analysis.get('major_network', {})
        if major_network:
            print(f"   基本ネットワーク指標:")
            print(f"     ノード数: {major_network.get('node_count', 0):,}")
            print(f"     エッジ数: {major_network.get('edge_count', 0):,}")
            print(f"     α指数: {major_network.get('alpha_index', 0):.1f}%")
            print(f"     道路密度: {major_network.get('road_density', 0):.1f} m/ha")
            print(f"     平均迂回率: {major_network.get('avg_circuity', 0):.2f}")
    
    # 軸線分析結果
    axial_analysis = results.get('axial_analysis', {})
    if axial_analysis and not axial_analysis.get('error'):
        network_metrics = axial_analysis.get('network_metrics', {})
        integration_stats = axial_analysis.get('integration_statistics', {})
        
        print(f"   軸線分析 (Axial Analysis):")
        if network_metrics:
            print(f"     軸線数: {network_metrics.get('axial_lines', 0)}")
            print(f"     軸線接続数: {network_metrics.get('axial_connections', 0)}")
            print(f"     格子度: {network_metrics.get('grid_axiality', 0):.3f}")
            print(f"     循環度: {network_metrics.get('axial_ringiness', 0):.3f}")
        
        if integration_stats:
            print(f"     Integration Value平均: {integration_stats.get('mean', 0):.3f}")
            print(f"     Integration Value標準偏差: {integration_stats.get('std', 0):.3f}")
    
    # 可視領域分析結果
    visibility_analysis = results.get('visibility_analysis', {})
    if visibility_analysis and not visibility_analysis.get('error'):
        visibility_field = visibility_analysis.get('visibility_field', {})
        field_stats = visibility_field.get('field_statistics', {})
        
        print(f"   可視領域分析 (Visibility Analysis):")
        if field_stats:
            print(f"     平均可視面積: {field_stats.get('mean_visible_area', 0):.1f} m²")
            print(f"     サンプリング点数: {field_stats.get('total_sampling_points', 0)}")
            print(f"     平均コンパクト性: {field_stats.get('mean_compactness', 0):.3f}")
    
    # 統合評価
    integrated_evaluation = results.get('integrated_evaluation', {})
    if integrated_evaluation and not integrated_evaluation.get('error'):
        print(f"   統合評価:")
        print(f"     回遊性スコア: {integrated_evaluation.get('connectivity_score', 0):.1f}/100")
        print(f"     アクセス性スコア: {integrated_evaluation.get('accessibility_score', 0):.1f}/100")
        print(f"     効率性スコア: {integrated_evaluation.get('efficiency_score', 0):.1f}/100")
        print(f"     総合スコア: {integrated_evaluation.get('overall_score', 0):.1f}/100")
        print(f"     評価レベル: {integrated_evaluation.get('evaluation_level', '評価不可')}")


def display_basic_results(results):
    """基本分析結果の表示"""
    print(f"\n📊 基本分析結果:")
    
    major_network = results.get('major_network', {})
    if major_network:
        print(f"   主要道路ネットワーク:")
        print(f"     ノード数: {major_network.get('node_count', 0):,}")
        print(f"     エッジ数: {major_network.get('edge_count', 0):,}")
        print(f"     α指数: {major_network.get('alpha_index', 0):.1f}%")
        print(f"     道路密度: {major_network.get('road_density', 0):.1f} m/ha")
        print(f"     平均迂回率: {major_network.get('avg_circuity', 0):.2f}")
    
    full_network = results.get('full_network', {})
    if full_network:
        print(f"   全道路ネットワーク:")
        print(f"     ノード数: {full_network.get('node_count', 0):,}")
        print(f"     エッジ数: {full_network.get('edge_count', 0):,}")


def export_results(analyzer, results, location, analysis_type):
    """分析結果のエクスポート"""
    try:
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        location_name = str(location).replace(',', '_').replace(' ', '_')
        
        # 基本エクスポート
        try:
            if hasattr(analyzer, 'export_results'):
                success = analyzer.export_results(
                    results,
                    str(output_dir / f"{analysis_type}_analysis_{location_name}.csv")
                )
                if success:
                    print(f"✅ 基本データエクスポート: {output_dir}")
            else:
                print("⚠️ エクスポート機能が利用できません")
        except Exception as e:
            print(f"⚠️ 基本エクスポートエラー: {e}")
        
        # JSON形式での保存
        try:
            import json
            json_path = output_dir / f"results_{analysis_type}_{location_name}.json"
            
            # JSONシリアライズ可能な形式に変換
            json_results = convert_to_json_serializable(results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ JSON結果保存: {json_path}")
            
        except Exception as e:
            print(f"⚠️ JSON保存エラー: {e}")
            
        # 詳細レポート生成（可能な場合）
        if analysis_type == "enhanced":
            try:
                report_path = output_dir / f"detailed_report_{location_name}.txt"
                report_content = generate_detailed_report(results, location)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                print(f"✅ 詳細レポート生成: {report_path}")
                
            except Exception as e:
                print(f"⚠️ 詳細レポート生成エラー: {e}")
            
    except Exception as e:
        print(f"⚠️ エクスポート処理エラー: {e}")
        logger.error(f"エクスポート処理エラー: {e}")


def generate_detailed_report(results, location):
    """詳細レポートを生成"""
    try:
        report_lines = [
            f"# {location} 拡張Space Syntax分析レポート",
            f"",
            f"## 分析概要",
            f"分析対象: {location}",
            f"分析面積: {results.get('area_ha', 0):.1f}ha",
            f"分析日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
        ]
        
        # 基本分析結果
        basic_analysis = results.get('basic_analysis', {})
        if basic_analysis:
            major_network = basic_analysis.get('major_network', {})
            if major_network:
                report_lines.extend([
                    "## 基本ネットワーク分析",
                    f"",
                    f"### 主要道路ネットワーク",
                    f"- ノード数: {major_network.get('node_count', 0):,}",
                    f"- エッジ数: {major_network.get('edge_count', 0):,}",
                    f"- α指数: {major_network.get('alpha_index', 0):.1f}%",
                    f"- β指数: {major_network.get('beta_index', 0):.2f}",
                    f"- γ指数: {major_network.get('gamma_index', 0):.1f}%",
                    f"- 道路密度: {major_network.get('road_density', 0):.1f} m/ha",
                    f"- 平均迂回率: {major_network.get('avg_circuity', 0):.2f}",
                    f"",
                ])
        
        # 軸線分析結果
        axial_analysis = results.get('axial_analysis', {})
        if axial_analysis and not axial_analysis.get('error'):
            network_metrics = axial_analysis.get('network_metrics', {})
            integration_stats = axial_analysis.get('integration_statistics', {})
            
            report_lines.extend([
                "## 軸線分析結果",
                f"",
                f"### 軸線ネットワーク基本統計",
                f"- 軸線数: {network_metrics.get('axial_lines', 0)}",
                f"- 軸線接続数: {network_metrics.get('axial_connections', 0)}",
                f"- アイランド数: {network_metrics.get('axial_islands', 0)}",
                f"",
                f"### 形態指標",
                f"- 格子度: {network_metrics.get('grid_axiality', 0):.3f}",
                f"- 循環度: {network_metrics.get('axial_ringiness', 0):.3f}",
                f"- 分節度: {network_metrics.get('axial_articulation', 0):.3f}",
                f"",
                f"### Integration Value統計",
                f"- 平均値: {integration_stats.get('mean', 0):.3f}",
                f"- 標準偏差: {integration_stats.get('std', 0):.3f}",
                f"- 最大値: {integration_stats.get('max', 0):.3f}",
                f"- 最小値: {integration_stats.get('min', 0):.3f}",
                f"",
            ])
        
        # 統合評価
        integrated_evaluation = results.get('integrated_evaluation', {})
        if integrated_evaluation and not integrated_evaluation.get('error'):
            report_lines.extend([
                "## 統合評価",
                f"",
                f"- 回遊性スコア: {integrated_evaluation.get('connectivity_score', 0):.1f}/100",
                f"- アクセス性スコア: {integrated_evaluation.get('accessibility_score', 0):.1f}/100", 
                f"- 効率性スコア: {integrated_evaluation.get('efficiency_score', 0):.1f}/100",
                f"- 総合スコア: {integrated_evaluation.get('overall_score', 0):.1f}/100",
                f"- 評価レベル: {integrated_evaluation.get('evaluation_level', '評価不可')}",
                f"",
            ])
        
        report_lines.extend([
            "---",
            "*本レポートは拡張space-syntax-analyzerにより自動生成されました*"
        ])
        
        return "\n".join(report_lines)
        
    except Exception as e:
        logger.error(f"詳細レポート生成エラー: {e}")
        return f"# {location} 分析レポート\n\nレポート生成中にエラーが発生しました: {e}"


def convert_to_json_serializable(obj):
    """オブジェクトをJSONシリアライズ可能な形式に変換"""
    import numpy as np
    
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key in ['axial_map', 'visibility_graph']:  # NetworkXオブジェクトをスキップ
                result[key] = f"NetworkX Graph with {len(value.nodes()) if hasattr(value, 'nodes') else 0} nodes"
            else:
                result[key] = convert_to_json_serializable(value)
        return result
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def try_visualization(analyzer, results, location, analysis_type):
    """可視化の試行"""
    try:
        output_dir = Path("demo_output")
        location_name = str(location).replace(',', '_').replace(' ', '_')
        
        # 基本可視化の試行
        if hasattr(analyzer, 'visualize'):
            try:
                vis_path = output_dir / f"basic_visualization_{location_name}.png"
                
                # ネットワークを取得
                major_net, full_net = analyzer.get_network(location, "both")
                
                success = analyzer.visualize(
                    major_net, full_net, results, str(vis_path)
                )
                if success:
                    print(f"✅ 基本可視化生成: {vis_path}")
                else:
                    print("⚠️ 基本可視化に失敗")
                    
            except Exception as e:
                print(f"⚠️ 基本可視化エラー: {e}")
                logger.error(f"基本可視化エラー: {e}")
            
    except Exception as e:
        print(f"⚠️ 可視化処理エラー: {e}")
        logger.error(f"可視化処理エラー: {e}")


def demo_axial_analysis_only():
    """軸線分析のみのデモ"""
    print("\n" + "="*50)
    print("🔗 軸線分析単体デモ")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.axial import AxialAnalyzer
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        # 基本アナライザーでネットワーク取得
        basic_analyzer = SpaceSyntaxAnalyzer()
        axial_analyzer = AxialAnalyzer()
        
        # 軸線分析実行
        test_location = "渋谷, 東京"
        print(f"📍 分析対象: {test_location}")
        
        # ネットワーク取得
        major_network, _ = basic_analyzer.get_network(test_location, "both")
        
        if major_network and major_network.number_of_nodes() > 0:
            print(f"✅ ネットワーク取得成功: {major_network.number_of_nodes()}ノード")
            
            # 軸線分析実行
            axial_results = axial_analyzer.calculate_axial_summary(major_network)
            
            print("\n📊 軸線分析結果:")
            network_metrics = axial_results.get('network_metrics', {})
            if network_metrics:
                print(f"   軸線数: {network_metrics.get('axial_lines', 0)}")
                print(f"   軸線接続数: {network_metrics.get('axial_connections', 0)}")
                print(f"   格子度: {network_metrics.get('grid_axiality', 0):.3f}")
                print(f"   循環度: {network_metrics.get('axial_ringiness', 0):.3f}")
            
            integration_stats = axial_results.get('integration_statistics', {})
            if integration_stats:
                print(f"   Integration Value統計:")
                print(f"     平均: {integration_stats.get('mean', 0):.3f}")
                print(f"     標準偏差: {integration_stats.get('std', 0):.3f}")
                print(f"     最大値: {integration_stats.get('max', 0):.3f}")
                print(f"     最小値: {integration_stats.get('min', 0):.3f}")
            
            # 結果保存
            output_dir = Path("demo_output")
            output_dir.mkdir(exist_ok=True)
            
            import json
            with open(output_dir / "axial_only_analysis.json", 'w', encoding='utf-8') as f:
                json_results = convert_to_json_serializable(axial_results)
                json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ 軸線分析結果保存: {output_dir / 'axial_only_analysis.json'}")
            
        else:
            print("❌ ネットワーク取得に失敗")
            
    except Exception as e:
        print(f"❌ 軸線分析デモエラー: {e}")
        logger.error(f"軸線分析デモエラー: {e}")


def demo_basic_analysis_fallback():
    """基本機能のみのフォールバックデモ"""
    print("\n" + "="*50)
    print("⚡ 基本機能デモ（フォールバック）")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        analyzer = SpaceSyntaxAnalyzer()
        
        # 基本分析実行
        test_location = "渋谷, 東京"
        print(f"📍 分析対象: {test_location}")
        
        results = analyzer.analyze_place(test_location)
        
        if not results.get('error', False):
            print("✅ 基本分析成功")
            display_basic_results(results)
            
            # レポート生成
            if hasattr(analyzer, 'generate_report'):
                report = analyzer.generate_report(results, f"{test_location} 基本分析結果")
                print(f"\n📄 分析レポート（抜粋）:")
                # レポートの最初の500文字のみ表示
                print(report[:500] + "..." if len(report) > 500 else report)
            
            # 結果保存
            output_dir = Path("demo_output")
            output_dir.mkdir(exist_ok=True)
            
            if hasattr(analyzer, 'export_results'):
                analyzer.export_results(results, str(output_dir / "basic_fallback_analysis.csv"))
                print(f"✅ 基本結果保存: {output_dir}")
            
        else:
            print(f"❌ 基本分析失敗: {results.get('error_message', '不明')}")
            
    except Exception as e:
        print(f"❌ 基本機能デモエラー: {e}")
        logger.error(f"基本機能デモエラー: {e}")


def check_enhanced_dependencies():
    """拡張機能の依存関係チェック"""
    print("🔍 拡張機能依存関係チェック中...")
    
    required_packages = [
        'osmnx',
        'networkx', 
        'pandas',
        'matplotlib',
        'numpy',
        'scipy',
        'shapely',
        'geopandas'
    ]
    
    optional_packages = [
        'scikit-learn',
        'plotly',
        'folium'
    ]
    
    missing_required = []
    missing_optional = []
    
    # 必須パッケージチェック
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (必須・未インストール)")
            missing_required.append(package)
    
    # オプションパッケージチェック
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} (オプション)")
        except ImportError:
            print(f"   ⚠️ {package} (オプション・未インストール)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ 必須パッケージ不足: {', '.join(missing_required)}")
        print(f"インストールコマンド: uv add {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️ オプション機能制限: {', '.join(missing_optional)}")
        print(f"フル機能使用には: uv add {' '.join(missing_optional)}")
    
    print("✅ 拡張機能の実行が可能です")
    return True


def main():
    """メイン関数"""
    print("🌟 最終修正版拡張 Space Syntax Analyzer デモンストレーション")
    print("="*80)
    
    # 依存関係チェック
    if not check_enhanced_dependencies():
        print("\n❌ 必要な依存関係が不足しています")
        return
    
    # 出力ディレクトリの作成
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir.absolute()}")
    
    # 自動実行モード判定
    auto_mode = os.getenv('DEMO_AUTO_MODE', 'false').lower() == 'true'
    
    if auto_mode:
        print(f"\n🤖 自動実行モード: 拡張包括分析を実行")
        demo_enhanced_comprehensive_analysis()
    else:
        # インタラクティブモード
        demos = [
            ("1", "拡張包括分析デモ（推奨）", demo_enhanced_comprehensive_analysis),
            ("2", "軸線分析単体デモ", demo_axial_analysis_only),
            ("3", "基本機能デモ", demo_basic_analysis_fallback),
            ("a", "全デモ実行", None),
        ]
        
        print(f"\n📋 利用可能なデモ:")
        for code, name, _ in demos:
            print(f"   {code}: {name}")
        
        choice = input(f"\n選択してください (1-3, a, q=終了): ").strip().lower()
        
        if choice == 'q':
            print("👋 デモンストレーション終了")
            return
        elif choice == 'a':
            print(f"\n🚀 全デモを順次実行します...")
            for code, name, func in demos[:-1]:  # 'a'以外の全て
                if func:
                    print(f"\n{'='*60}")
                    print(f"▶️ {name} 開始")
                    print(f"{'='*60}")
                    func()
                    print(f"✅ {name} 完了")
                    time.sleep(2)  # 少し間隔を開ける
        else:
            # 個別デモの実行
            for code, name, func in demos:
                if choice == code and func:
                    print(f"\n{'='*60}")
                    print(f"▶️ {name} 開始")
                    print(f"{'='*60}")
                    func()
                    print(f"✅ {name} 完了")
                    break
            else:
                print(f"❌ 無効な選択: {choice}")
    
    print(f"\n🎉 最終修正版拡張デモンストレーション完了!")
    print(f"📁 結果ファイルは {output_dir.absolute()} に保存されています")
    print(f"")
    print(f"📚 生成された主要ファイル:")
    print(f"   - enhanced_analysis_*.csv: 拡張分析データ")
    print(f"   - results_*.json: 詳細分析結果")
    print(f"   - detailed_report_*.txt: 拡張分析レポート")
    print(f"   - *_visualization_*.png: 可視化画像")
    print(f"   - axial_only_analysis.json: 軸線分析詳細データ")
    print(f"")
    print(f"💡 最終修正版の特徴:")
    print(f"   • 基底クラスの正確なシグネチャに対応")
    print(f"   • カスタム拡張アナライザーによる継承問題解決")
    print(f"   • 軸線分析・可視領域分析の統合実装")
    print(f"   • エラー時の適切なフォールバック機能")
    print(f"   • 包括的な分析レポート自動生成")


if __name__ == "__main__":
    main()