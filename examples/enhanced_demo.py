# 最強のフォント警告抑制
import logging
import warnings

# matplotlib関連の全警告を抑制
warnings.filterwarnings('ignore', message='findfont: Font family.*not found')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
warnings.filterwarnings('ignore', message='.*font.*not found.*')

# matplotlibロガーの警告レベルを上げる
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# matplotlib設定（インポート前）
import matplotlib

matplotlib.use('Agg')  # GUIなしバックエンド
import matplotlib.pyplot as plt

# 日本語フォント設定
try:
    import japanize_matplotlib

    # japanize_matplotlibが自動的に日本語フォントを設定
    print("✅ 日本語フォント設定完了")
    JAPANESE_FONT_AVAILABLE = True
except ImportError:
    # フォールバック：英語フォント
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    print("⚠️ 日本語フォントが利用できません。英語フォントを使用します。")
    JAPANESE_FONT_AVAILABLE = False

plt.rcParams["axes.unicode_minus"] = False

# その他のインポート
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
from shapely.geometry.base import BaseGeometry

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# グローバル変数：分析結果保存用
ANALYSIS_RESULTS = {}
CITY_NAMES_JP = {
    "Matsumoto, Nagano, Japan": "松本市",
    "Nagano City, Nagano, Japan": "長野市",
    "Ueda, Nagano, Japan": "上田市"
}


def demo_enhanced_comprehensive_analysis():
    """最適化版包括的分析のデモンストレーション"""
    print("="*60)
    print("🚀 最適化版 Space Syntax 包括的分析デモンストレーション")
    print("="*60)
    
    try:
        # まず基本アナライザーをテスト
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer

        # 拡張アナライザーの利用可能性をテスト
        enhanced_available = False
        try:
            # カスタムEnhancedSpaceSyntaxAnalyzerクラスを作成
            analyzer = create_optimized_enhanced_analyzer()
            enhanced_available = True
            print("✅ 最適化拡張アナライザー初期化成功")
            analysis_type = "enhanced_optimized"
        except Exception as e:
            print(f"⚠️ 拡張アナライザー初期化エラー: {e}")
            print("🔄 基本アナライザーを使用")
            analyzer = SpaceSyntaxAnalyzer()
            analysis_type = "basic"
        
        # 長野県の代表都市に変更
        test_locations = [
            "Matsumoto, Nagano, Japan",      # 松本市
            "Nagano City, Nagano, Japan",    # 長野市
            "Ueda, Nagano, Japan"           # 上田市
        ]
        
        successful_analyses = []
        
        for location in test_locations:
            location_jp = CITY_NAMES_JP.get(location, location)
            print(f"\n📍 分析試行: {location_jp} ({location})")
            
            try:
                start_time = time.time()
                
                if enhanced_available:
                    # 最適化拡張分析を実行
                    results = perform_optimized_enhanced_analysis(analyzer, location)
                else:
                    # 基本分析を実行
                    results = analyzer.analyze_place(location)
                
                end_time = time.time()
                
                # エラーチェック（型安全性を確保）
                if isinstance(results, dict) and results.get('error', False):
                    print(f"❌ 分析エラー: {results.get('error_message', '不明')}")
                    continue
                elif not isinstance(results, dict):
                    print(f"❌ 結果型エラー: 予期しない型 {type(results)}")
                    logger.error(f"結果型エラー: {type(results)} - {results}")
                    continue
                
                # 成功
                execution_time = end_time - start_time
                successful_analyses.append((location, results, analysis_type, execution_time))
                
                # グローバル変数に保存（レポート用）
                ANALYSIS_RESULTS[location] = {
                    'results': results,
                    'analysis_type': analysis_type,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                }
                
                print(f"✅ 分析成功! (実行時間: {execution_time:.1f}秒)")
                
            except Exception as e:
                print(f"❌ 例外発生: {e}")
                logger.error(f"分析例外 ({location}): {e}")
                continue
        
        if successful_analyses:
            print(f"\n🎯 分析完了: {len(successful_analyses)}都市の分析に成功")
            
            # 各都市の結果表示
            for location, results, analysis_type, execution_time in successful_analyses:
                location_jp = CITY_NAMES_JP.get(location, location)
                print(f"\n📊 {location_jp}の分析結果:")
                
                if "enhanced" in analysis_type:
                    display_enhanced_results(results)
                else:
                    display_basic_results(results)
                
                # 結果のエクスポート
                export_results(analyzer, results, location, analysis_type)
                
                # 可視化（可能な場合）
                try_visualization(analyzer, results, location, analysis_type)
            
            # 比較分析レポート生成
            generate_comparative_report(successful_analyses)
            
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


def create_optimized_enhanced_analyzer():
    """最適化カスタム拡張アナライザーを作成"""
    import networkx as nx  # NetworkXのインポートを追加
    import pandas as pd  # Pandasのインポートを追加

    from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
    from space_syntax_analyzer.core.axial import AxialAnalyzer
    from space_syntax_analyzer.core.visibility import VisibilityAnalyzer

    # 基本アナライザーを拡張
    class OptimizedEnhancedAnalyzer(SpaceSyntaxAnalyzer):
        def __init__(self):
            # 基底クラスの正しいシグネチャで初期化
            super().__init__(network_type="drive", width_threshold=4.0)
            
            # 最適化された拡張機能を追加
            self.enable_axial_analysis = True
            self.enable_visibility_analysis = True
            
            # 最適化パラメータ
            self.max_intersections = 30  # 交差点数制限
            self.max_sampling_points = 50  # サンプリング点数制限
            self.visibility_radius = 80.0  # 可視半径短縮
            
            # 拡張アナライザーの初期化
            self.axial_analyzer = AxialAnalyzer()
            self.visibility_analyzer = VisibilityAnalyzer(
                visibility_radius=self.visibility_radius,
                max_intersections=self.max_intersections
            )
        
        def analyze_comprehensive(self, location, return_networks=False, analysis_level="global"):
            """最適化包括的分析の実行"""
            try:
                logger.info(f"最適化包括的分析開始: {location}")
                
                # 基本分析を実行
                basic_result = self.analyze_place(location, return_networks=True)
                
                # 結果の型チェックと正規化
                if isinstance(basic_result, tuple):
                    results, networks = basic_result
                    major_network, full_network = networks if networks else (None, None)
                else:
                    results = basic_result
                    major_network = full_network = None
                
                # エラーチェック
                if isinstance(results, dict) and results.get('error', False):
                    return results
                
                # 包括的分析結果の構築
                comprehensive_results = {
                    "basic_analysis": results,
                }
                
                # 軸線分析（並列実行対応）
                if self.enable_axial_analysis and major_network:
                    logger.info("最適化軸線分析開始")
                    try:
                        axial_results = self._perform_optimized_axial_analysis(
                            major_network, analysis_level
                        )
                        comprehensive_results["axial_analysis"] = axial_results
                        logger.info("軸線分析完了")
                    except Exception as e:
                        logger.warning(f"軸線分析スキップ: {e}")
                        comprehensive_results["axial_analysis"] = {"error": str(e)}
                
                # 最適化可視領域分析
                if self.enable_visibility_analysis and major_network:
                    logger.info("最適化可視領域分析開始")
                    try:
                        visibility_results = self._perform_optimized_visibility_analysis(major_network)
                        comprehensive_results["visibility_analysis"] = visibility_results
                        logger.info("可視領域分析完了")
                    except Exception as e:
                        logger.warning(f"可視領域分析スキップ: {e}")
                        comprehensive_results["visibility_analysis"] = {"error": str(e)}
                
                # 統合評価の生成
                try:
                    integrated_evaluation = self._generate_integrated_evaluation(comprehensive_results)
                    comprehensive_results["integrated_evaluation"] = integrated_evaluation
                except Exception as e:
                    logger.warning(f"統合評価生成エラー: {e}")
                    comprehensive_results["integrated_evaluation"] = {"error": str(e)}
                
                logger.info("最適化包括的分析完了")
                
                # 戻り値の型を統一
                if return_networks:
                    return comprehensive_results, (major_network, full_network)
                else:
                    return comprehensive_results
                
            except Exception as e:
                logger.error(f"最適化包括的分析エラー: {e}")
                return {
                    "error": True,
                    "error_message": str(e),
                    "analysis_type": "optimized_comprehensive"
                }
        
        def _perform_optimized_axial_analysis(self, network, analysis_level="global"):
            """最適化軸線分析を実行"""
            try:
                import networkx as nx  # ローカルインポート追加

                # 軸線分析の実行
                axial_results = self.axial_analyzer.calculate_axial_summary(network)
                
                # 分析レベルに応じた追加計算（簡略化）
                axial_map = axial_results.get("axial_map", nx.Graph())
                
                if analysis_level in ["global", "both"] and axial_map.number_of_nodes() > 0:
                    global_integration = self.axial_analyzer.analyze_global_integration(axial_map)
                    axial_results["global_integration"] = global_integration
                
                if analysis_level in ["local", "both"] and axial_map.number_of_nodes() > 0:
                    local_integration = self.axial_analyzer.analyze_local_integration(axial_map)
                    axial_results["local_integration"] = local_integration
                
                return axial_results
                
            except Exception as e:
                logger.warning(f"最適化軸線分析実行エラー: {e}")
                return {"error": str(e)}
        
        def _perform_optimized_visibility_analysis(self, network):
            """最適化可視領域分析を実行"""
            try:
                # 最適化可視領域フィールド分析
                visibility_field = self.visibility_analyzer.analyze_visibility_field(
                    network, 
                    sampling_distance=40.0,  # サンプリング間隔拡大
                    max_points=self.max_sampling_points  # 点数制限
                )
                
                # 最適化視覚的接続性分析
                visual_connectivity = self.visibility_analyzer.analyze_visual_connectivity(network)
                
                return {
                    "visibility_field": visibility_field,
                    "visual_connectivity": visual_connectivity,
                }
                
            except Exception as e:
                logger.warning(f"最適化可視領域分析実行エラー: {e}")
                return {"error": str(e)}
        
        def _generate_integrated_evaluation(self, results):
            """統合評価を生成"""
            try:
                import pandas as pd  # ローカルインポート追加
                
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
                accessibility_score = min(road_density * 5, 100)
                
                # 効率性スコア
                circuity = major_network.get("avg_circuity", 1.0)
                efficiency_score = max(0, min((2.0 - circuity) / 1.0 * 100, 100))
                
                # 総合スコア
                overall_score = (connectivity_score + accessibility_score + efficiency_score) / 3
                
                # 評価レベルの判定
                if overall_score >= 80:
                    evaluation_level = "A - 優秀"
                elif overall_score >= 65:
                    evaluation_level = "B - 良好"
                elif overall_score >= 50:
                    evaluation_level = "C - 標準"
                elif overall_score >= 35:
                    evaluation_level = "D - 要改善"
                else:
                    evaluation_level = "E - 大幅改善必要"
                
                return {
                    "connectivity_score": connectivity_score,
                    "accessibility_score": accessibility_score,
                    "efficiency_score": efficiency_score,
                    "overall_score": overall_score,
                    "evaluation_level": evaluation_level,
                    "analysis_timestamp": pd.Timestamp.now().isoformat(),
                }
                
            except Exception as e:
                logger.warning(f"統合評価生成エラー: {e}")
                return {"error": str(e)}
    
    return OptimizedEnhancedAnalyzer()


def perform_optimized_enhanced_analysis(analyzer, location):
    """最適化拡張分析を実行"""
    try:
        result = analyzer.analyze_comprehensive(
            location, 
            return_networks=True, 
            analysis_level="global"  # globalのみで高速化
        )
        
        # タプルが返された場合の処理
        if isinstance(result, tuple):
            comprehensive_results, networks = result
            return comprehensive_results
        else:
            # 辞書が直接返された場合
            return result
            
    except Exception as e:
        logger.error(f"最適化拡張分析エラー: {e}")
        return {
            "error": True,
            "error_message": str(e)
        }


def display_enhanced_results(results):
    """最適化拡張分析結果の表示"""
    print(f"\n📊 最適化拡張分析結果:")
    
    # 基本分析結果
    basic_analysis = results.get('basic_analysis', {})
    if basic_analysis:
        major_network = basic_analysis.get('major_network', {})
        if major_network:
            print(f"   基本ネットワーク指標:")
            print(f"     ノード数: {major_network.get('node_count', 0):,}")
            print(f"     エッジ数: {major_network.get('edge_count', 0):,}")
            print(f"     α指数: {major_network.get('alpha_index', 0):.2f}")
            print(f"     γ指数: {major_network.get('gamma_index', 0):.2f}")
            print(f"     道路密度: {major_network.get('road_density', 0):.2f} km/km²")
            print(f"     平均迂回率: {major_network.get('avg_circuity', 0):.2f}")
    
    # 軸線分析結果
    axial_analysis = results.get('axial_analysis', {})
    if axial_analysis and not axial_analysis.get('error'):
        network_metrics = axial_analysis.get('network_metrics', {})
        integration_stats = axial_analysis.get('integration_statistics', {})
        
        print(f"   軸線分析:")
        if network_metrics:
            print(f"     軸線数: {network_metrics.get('axial_lines', 0)}")
            print(f"     軸線接続数: {network_metrics.get('axial_connections', 0)}")
            print(f"     格子度: {network_metrics.get('grid_axiality', 0):.3f}")
        
        if integration_stats:
            print(f"     統合値平均: {integration_stats.get('mean', 0):.3f}")
    
    # 可視領域分析結果
    visibility_analysis = results.get('visibility_analysis', {})
    if visibility_analysis and not visibility_analysis.get('error'):
        visibility_field = visibility_analysis.get('visibility_field', {})
        field_stats = visibility_field.get('field_statistics', {})
        visual_connectivity = visibility_analysis.get('visual_connectivity', {})
        
        print(f"   可視領域分析:")
        if field_stats:
            print(f"     平均可視面積: {field_stats.get('mean_visible_area', 0):.1f} m²")
            print(f"     サンプリング点数: {field_stats.get('total_sampling_points', 0)}")
        
        if visual_connectivity:
            network_metrics = visual_connectivity.get('network_metrics', {})
            if network_metrics:
                print(f"     視覚的接続ノード数: {network_metrics.get('visual_nodes', 0)}")
                print(f"     視覚的接続エッジ数: {network_metrics.get('visual_edges', 0)}")
    
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
        print(f"     α指数: {major_network.get('alpha_index', 0):.2f}")
        print(f"     γ指数: {major_network.get('gamma_index', 0):.2f}")
        print(f"     道路密度: {major_network.get('road_density', 0):.2f} km/km²")


def export_results(analyzer, results, location, analysis_type):
    """結果のエクスポート（タプルキー対応版）"""
    try:
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{location.replace(',', '_').replace(' ', '_')}_{analysis_type}_{timestamp}.json"
        
        import json

        # 段階的変換処理
        try:
            print(f"   🔍 結果変換開始...")
            serializable_results = convert_to_serializable(results)
            print(f"   ✅ 結果変換完了")
            
            # JSONテスト
            json_test = json.dumps(serializable_results)
            print(f"   ✅ JSON変換テスト成功")
            
        except Exception as convert_error:
            print(f"   ⚠️ 変換エラー: {convert_error}")
            # より安全なフォールバック処理
            serializable_results = create_safe_fallback_data(results, location, analysis_type, str(convert_error))
        
        # ファイル保存
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 結果保存: {filename}")
        
    except Exception as e:
        logger.warning(f"結果エクスポートエラー: {e}")
        print(f"   ⚠️ エクスポート処理でエラーが発生しましたが、分析は正常に完了しています")


def create_safe_fallback_data(results, location, analysis_type, error_msg):
    """安全なフォールバックデータを作成"""
    safe_data = {
        "metadata": {
            "location": location,
            "analysis_type": analysis_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "export_status": "partial_due_to_serialization_error",
            "error_message": error_msg
        }
    }
    
    # 基本的な数値データのみ抽出
    def extract_safe_data(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, (str, int, float, bool)) or key is None:
                    new_path = f"{path}.{key}" if path else str(key)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_data[new_path] = value
                    elif isinstance(value, dict):
                        extract_safe_data(value, new_path)
                    elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value[:5]):
                        safe_data[new_path] = value[:5]  # 最初の5要素のみ
    
    try:
        extract_safe_data(results)
    except Exception:
        safe_data["extraction_error"] = "Failed to extract safe data"
    
    return safe_data


def convert_to_serializable(obj):
    """オブジェクトをJSON serializable に変換"""
    import networkx as nx  # ローカルインポート追加
    import numpy as np
    from shapely.geometry.base import BaseGeometry
    
    if isinstance(obj, dict):
        # 辞書のキーがタプルの場合は文字列に変換
        converted_dict = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                # タプルキーを文字列に変換
                key_str = f"{k[0]}_{k[1]}" if len(k) == 2 else "_".join(map(str, k))
                converted_dict[key_str] = convert_to_serializable(v)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                converted_dict[k] = convert_to_serializable(v)
            else:
                # その他の型のキーは文字列に変換
                converted_dict[str(k)] = convert_to_serializable(v)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, nx.Graph):
        return {"type": "networkx.Graph", "nodes": len(obj.nodes()), "edges": len(obj.edges())}
    elif isinstance(obj, BaseGeometry):
        # Shapely幾何学オブジェクトの処理
        return {
            "type": "shapely_geometry",
            "geometry_type": obj.geom_type,
            "area": getattr(obj, 'area', 0),
            "length": getattr(obj, 'length', 0),
            "bounds": list(obj.bounds) if hasattr(obj, 'bounds') else [],
            "is_valid": obj.is_valid if hasattr(obj, 'is_valid') else True
        }
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return {"type": str(type(obj)), "value": str(obj)}
    elif callable(obj):
        return {"type": "callable", "name": getattr(obj, '__name__', 'unknown')}
    else:
        try:
            # 基本的なJSON serializableかチェック
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return {"type": str(type(obj)), "value": str(obj)}


def create_comprehensive_charts(results, location):
    """包括的で意味のあるチャートセットを作成（日本語版）"""
    try:
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np

        # matplotlib警告を抑制
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # 日本語都市名を取得
        location_jp = CITY_NAMES_JP.get(location, location)
        
        # 基本分析結果からデータを抽出
        basic_analysis = results.get('basic_analysis', {})
        major_network = basic_analysis.get('major_network', {})
        axial_analysis = results.get('axial_analysis', {})
        visibility_analysis = results.get('visibility_analysis', {})
        integrated_evaluation = results.get('integrated_evaluation', {})
        
        if not major_network:
            return
        
        # 出力ディレクトリ
        output_dir = Path("demo_output")
        base_filename = f"analysis_{location.replace(',', '_').replace(' ', '_')}"
        
        # 1. ネットワーク基本指標チャート
        create_network_metrics_chart_jp(major_network, location_jp, output_dir, base_filename)
        
        # 2. Space Syntax指標チャート
        create_space_syntax_chart_jp(major_network, location_jp, output_dir, base_filename)
        
        # 3. 軸線分析結果チャート
        if axial_analysis and not axial_analysis.get('error'):
            create_axial_analysis_chart_jp(axial_analysis, location_jp, output_dir, base_filename)
        
        # 4. 統合評価チャート
        if integrated_evaluation and not integrated_evaluation.get('error'):
            create_integrated_evaluation_chart_jp(integrated_evaluation, location_jp, output_dir, base_filename)
        
        # 5. 包括的ダッシュボード
        create_comprehensive_dashboard_jp(results, location_jp, output_dir, base_filename)
        
        print(f"   📊 包括的チャートセット生成完了 ({location_jp})")
        
    except Exception as e:
        logger.debug(f"包括的チャート作成エラー: {e}")
        pass


def create_network_metrics_chart_jp(major_network, location_jp, output_dir, base_filename):
    """ネットワーク基本指標チャート（日本語版）"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{location_jp} - ネットワーク基本指標', fontsize=14, fontweight='bold')
        
        # 1. ノード・エッジ数
        counts = [major_network.get('node_count', 0), major_network.get('edge_count', 0)]
        labels = ['ノード数', 'エッジ数']
        colors = ['#2E86AB', '#A23B72']
        
        ax1.bar(labels, counts, color=colors)
        ax1.set_title('ネットワーク規模', fontweight='bold')
        ax1.set_ylabel('数量')
        for i, v in enumerate(counts):
            ax1.text(i, v + max(counts)*0.01, f'{v:,}', ha='center', va='bottom')
        
        # 2. 接続性指標
        connectivity = {
            '平均次数': major_network.get('avg_degree', 0),
            '最大次数': major_network.get('max_degree', 0),
            '密度×1000': major_network.get('density', 0) * 1000  # スケール調整
        }
        
        ax2.bar(connectivity.keys(), connectivity.values(), color='#F18F01')
        ax2.set_title('接続性指標', fontweight='bold')
        ax2.set_ylabel('値')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Space Syntax基本指標
        syntax_metrics = {
            'α指数': major_network.get('alpha_index', 0),
            'β指数': major_network.get('beta_index', 0),
            'γ指数': major_network.get('gamma_index', 0)
        }
        
        ax3.bar(syntax_metrics.keys(), syntax_metrics.values(), color='#C73E1D')
        ax3.set_title('Space Syntax 指標', fontweight='bold')
        ax3.set_ylabel('値')
        
        # 4. 効率性指標
        efficiency_data = {
            '道路密度\n(km/km²)': major_network.get('road_density', 0),
            '平均迂回率': major_network.get('avg_circuity', 0),
            '連結成分数': major_network.get('num_components', 0)
        }
        
        ax4.bar(efficiency_data.keys(), efficiency_data.values(), color='#3F7D20')
        ax4.set_title('効率性指標', fontweight='bold')
        ax4.set_ylabel('値')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_file = output_dir / f"{base_filename}_network_metrics.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 ネットワーク指標チャート: {chart_file.name}")
        
    except Exception as e:
        logger.debug(f"ネットワーク指標チャート作成エラー: {e}")


def create_space_syntax_chart_jp(major_network, location_jp, output_dir, base_filename):
    """Space Syntax専用チャート（日本語版）"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{location_jp} - Space Syntax 分析', fontsize=14, fontweight='bold')
        
        # 1. 指標値の棒グラフ
        indices = {
            'α指数': major_network.get('alpha_index', 0),
            'β指数': major_network.get('beta_index', 0), 
            'γ指数': major_network.get('gamma_index', 0)
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax1.bar(indices.keys(), indices.values(), color=colors)
        ax1.set_title('Space Syntax 指標値', fontweight='bold')
        ax1.set_ylabel('指標値')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, indices.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. 理論的基準との比較（円グラフ）
        categories = ['α指数\n(循環性)', 'β指数\n(複雑性)', 'γ指数\n(接続性)']
        values = [
            min(major_network.get('alpha_index', 0) / 50 * 100, 100),  # 正規化
            min(major_network.get('beta_index', 0) / 3 * 100, 100),
            min(major_network.get('gamma_index', 0) / 1 * 100, 100)
        ]
        
        # 円グラフとして表示
        ax2.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, 
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('相対的分布', fontweight='bold')
        
        plt.tight_layout()
        chart_file = output_dir / f"{base_filename}_space_syntax.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Space Syntax チャート: {chart_file.name}")
        
    except Exception as e:
        logger.debug(f"Space Syntaxチャート作成エラー: {e}")


def create_axial_analysis_chart_jp(axial_analysis, location_jp, output_dir, base_filename):
    """軸線分析専用チャート（日本語版）"""
    try:
        network_metrics = axial_analysis.get('network_metrics', {})
        integration_stats = axial_analysis.get('integration_statistics', {})
        
        if not network_metrics and not integration_stats:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{location_jp} - 軸線分析結果', fontsize=14, fontweight='bold')
        
        # 1. 軸線ネットワーク指標
        if network_metrics:
            axial_data = {
                '軸線数': network_metrics.get('axial_lines', 0),
                '接続数': network_metrics.get('axial_connections', 0),
                '孤立数': network_metrics.get('axial_islands', 0)
            }
            
            ax1.bar(axial_data.keys(), axial_data.values(), color='#8E44AD')
            ax1.set_title('軸線ネットワーク構造', fontweight='bold')
            ax1.set_ylabel('数量')
            ax1.tick_params(axis='x', rotation=45)
            
            # 値をバーの上に表示
            for i, (k, v) in enumerate(axial_data.items()):
                ax1.text(i, v + max(axial_data.values())*0.01, f'{v:,}', ha='center', va='bottom')
        
        # 2. 形態指標
        if network_metrics:
            morphology = {
                '格子軸性': network_metrics.get('grid_axiality', 0) * 1000,  # スケール調整
                '軸線環状性': network_metrics.get('axial_ringiness', 0),
                '関節性': network_metrics.get('axial_articulation', 0) * 100  # パーセント表示
            }
            
            ax2.bar(morphology.keys(), morphology.values(), color='#27AE60')
            ax2.set_title('形態学的指標', fontweight='bold')
            ax2.set_ylabel('正規化値')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. 統合値統計
        if integration_stats:
            stats_data = {
                '平均': integration_stats.get('mean', 0),
                '標準偏差': integration_stats.get('std', 0),
                '中央値': integration_stats.get('median', 0)
            }
            
            ax3.bar(stats_data.keys(), stats_data.values(), color='#E74C3C')
            ax3.set_title('統合値統計', fontweight='bold')
            ax3.set_ylabel('統合値')
            
            # 値をバーの上に表示
            for i, (k, v) in enumerate(stats_data.items()):
                ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. 統合値分布（理論的分布）
        if integration_stats:
            mean = integration_stats.get('mean', 0)
            std = integration_stats.get('std', 0)
            min_val = integration_stats.get('min', 0)
            max_val = integration_stats.get('max', 0)
            
            # 理論的分布を表示
            x = np.linspace(min_val, max_val, 100)
            y = np.exp(-0.5 * ((x - mean) / std) ** 2) if std > 0 else np.ones_like(x)
            
            ax4.plot(x, y, 'b-', linewidth=2, label='分布形状')
            ax4.axvline(mean, color='r', linestyle='--', label=f'平均: {mean:.3f}')
            ax4.axvline(min_val, color='g', linestyle=':', label=f'最小: {min_val:.3f}')
            ax4.axvline(max_val, color='g', linestyle=':', label=f'最大: {max_val:.3f}')
            
            ax4.set_title('統合値分布', fontweight='bold')
            ax4.set_xlabel('統合値')
            ax4.set_ylabel('相対頻度')
            ax4.legend()
        
        plt.tight_layout()
        chart_file = output_dir / f"{base_filename}_axial_analysis.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 軸線分析チャート: {chart_file.name}")
        
    except Exception as e:
        logger.debug(f"軸線分析チャート作成エラー: {e}")


def create_integrated_evaluation_chart_jp(evaluation, location_jp, output_dir, base_filename):
    """統合評価チャート（日本語版）"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{location_jp} - 統合評価', fontsize=14, fontweight='bold')
        
        # 1. スコア比較
        scores = {
            '回遊性': evaluation.get('connectivity_score', 0),
            'アクセス性': evaluation.get('accessibility_score', 0),
            '効率性': evaluation.get('efficiency_score', 0),
            '総合': evaluation.get('overall_score', 0)
        }
        
        colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
        bars = ax1.bar(scores.keys(), scores.values(), color=colors)
        ax1.set_title('パフォーマンススコア', fontweight='bold')
        ax1.set_ylabel('スコア (0-100)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # スコアバーに値を表示
        for bar, value in zip(bars, scores.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. レーダーチャート
        angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
        values = list(scores.values())
        
        # レーダーチャートを円形に閉じる
        angles += angles[:1]
        values += values[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='#E74C3C')
        ax2.fill(angles, values, alpha=0.25, color='#E74C3C')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(scores.keys())
        ax2.set_ylim(0, 100)
        ax2.set_title('パフォーマンス レーダー', fontweight='bold', pad=20)
        
        # 評価レベルを表示
        evaluation_level = evaluation.get('evaluation_level', '不明')
        fig.text(0.5, 0.02, f'評価レベル: {evaluation_level}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        chart_file = output_dir / f"{base_filename}_evaluation.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 統合評価チャート: {chart_file.name}")
        
    except Exception as e:
        logger.debug(f"統合評価チャート作成エラー: {e}")


def create_comprehensive_dashboard_jp(results, location_jp, output_dir, base_filename):
    """包括的ダッシュボード（日本語版）"""
    try:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle(f'{location_jp} - 包括的 Space Syntax 分析ダッシュボード', 
                     fontsize=16, fontweight='bold')
        
        basic_analysis = results.get('basic_analysis', {})
        major_network = basic_analysis.get('major_network', {})
        axial_analysis = results.get('axial_analysis', {})
        integrated_evaluation = results.get('integrated_evaluation', {})
        
        # 1. ネットワーク概要 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        network_summary = {
            'ノード': major_network.get('node_count', 0),
            'エッジ': major_network.get('edge_count', 0)
        }
        ax1.bar(network_summary.keys(), network_summary.values(), color=['#3498DB', '#2ECC71'])
        ax1.set_title('ネットワーク規模', fontweight='bold')
        ax1.set_ylabel('数量')
        
        # 2. Space Syntax指標 (右上)
        ax2 = fig.add_subplot(gs[0, 2:])
        syntax_indices = {
            'α指数': major_network.get('alpha_index', 0),
            'β指数': major_network.get('beta_index', 0),
            'γ指数': major_network.get('gamma_index', 0)
        }
        ax2.bar(syntax_indices.keys(), syntax_indices.values(), color=['#E74C3C', '#F39C12', '#9B59B6'])
        ax2.set_title('Space Syntax 指標', fontweight='bold')
        ax2.set_ylabel('指標値')
        
        # 3. 軸線分析 (中段左)
        ax3 = fig.add_subplot(gs[1, :2])
        if axial_analysis and not axial_analysis.get('error'):
            network_metrics = axial_analysis.get('network_metrics', {})
            axial_data = {
                '軸線': network_metrics.get('axial_lines', 0),
                '接続': network_metrics.get('axial_connections', 0)
            }
            ax3.bar(axial_data.keys(), axial_data.values(), color=['#8E44AD', '#27AE60'])
            ax3.set_title('軸線分析', fontweight='bold')
            ax3.set_ylabel('数量')
        else:
            ax3.text(0.5, 0.5, '軸線分析\n利用不可', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        # 4. 統合値統計 (中段右)
        ax4 = fig.add_subplot(gs[1, 2:])
        if axial_analysis and not axial_analysis.get('error'):
            integration_stats = axial_analysis.get('integration_statistics', {})
            if integration_stats:
                stats = {
                    '平均': integration_stats.get('mean', 0),
                    '標準偏差': integration_stats.get('std', 0),
                    '最大': integration_stats.get('max', 0)
                }
                ax4.bar(stats.keys(), stats.values(), color=['#E67E22', '#D35400', '#C0392B'])
                ax4.set_title('統合値統計', fontweight='bold')
                ax4.set_ylabel('値')
        
        # 5. 統合評価 (下段)
        ax5 = fig.add_subplot(gs[2, :])
        if integrated_evaluation and not integrated_evaluation.get('error'):
            eval_scores = {
                '回遊性': integrated_evaluation.get('connectivity_score', 0),
                'アクセス性': integrated_evaluation.get('accessibility_score', 0),
                '効率性': integrated_evaluation.get('efficiency_score', 0),
                '総合': integrated_evaluation.get('overall_score', 0)
            }
            
            colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
            bars = ax5.bar(eval_scores.keys(), eval_scores.values(), color=colors)
            ax5.set_title('統合評価スコア', fontweight='bold')
            ax5.set_ylabel('スコア (0-100)')
            ax5.set_ylim(0, 100)
            
            # 評価レベルをテキストで表示
            evaluation_level = integrated_evaluation.get('evaluation_level', '不明')
            ax5.text(0.98, 0.95, f'レベル: {evaluation_level}', transform=ax5.transAxes, 
                    ha='right', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        chart_file = output_dir / f"{base_filename}_dashboard.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 包括的ダッシュボード: {chart_file.name}")
        
    except Exception as e:
        logger.debug(f"ダッシュボード作成エラー: {e}")


def try_visualization(analyzer, results, location, analysis_type):
    """改善された可視化の試行"""
    try:
        print(f"   📈 可視化生成中...")
        
        # 包括的チャートセットを作成
        create_comprehensive_charts(results, location)
        
        print(f"   📈 可視化完了")
        
    except Exception as e:
        logger.warning(f"可視化エラー: {e}")
        print(f"   ⚠️ 可視化をスキップしました")


def generate_comparative_report(successful_analyses):
    """都市間比較分析レポートをマークダウン形式で生成"""
    try:
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"comparative_analysis_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # レポートヘッダー
            f.write("# 長野県主要都市 Space Syntax 比較分析レポート\n\n")
            f.write(f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            f.write("## 📋 分析対象都市\n\n")
            
            analyzed_cities = []
            for location, results, analysis_type, execution_time in successful_analyses:
                city_jp = CITY_NAMES_JP.get(location, location)
                analyzed_cities.append(city_jp)
                f.write(f"- **{city_jp}** ({location})\n")
            
            f.write(f"\n**分析都市数**: {len(analyzed_cities)}都市\n\n")
            
            # 実行概要
            f.write("## ⚡ 実行概要\n\n")
            f.write("| 都市 | 分析タイプ | 実行時間 | ステータス |\n")
            f.write("|------|------------|----------|------------|\n")
            
            for location, results, analysis_type, execution_time in successful_analyses:
                city_jp = CITY_NAMES_JP.get(location, location)
                status = "✅ 成功" if not results.get('error', False) else "❌ エラー"
                f.write(f"| {city_jp} | {analysis_type} | {execution_time:.1f}秒 | {status} |\n")
            
            # 基本ネットワーク指標比較
            f.write("\n## 📊 基本ネットワーク指標比較\n\n")
            f.write("### ネットワーク規模\n\n")
            f.write("| 都市 | ノード数 | エッジ数 | 平均次数 | 最大次数 | 密度 |\n")
            f.write("|------|----------|----------|----------|----------|------|\n")
            
            # データ収集と表示
            network_data = {}
            for location, results, analysis_type, execution_time in successful_analyses:
                city_jp = CITY_NAMES_JP.get(location, location)
                basic_analysis = results.get('basic_analysis', {})
                major_network = basic_analysis.get('major_network', {})
                
                network_data[city_jp] = {
                    'node_count': major_network.get('node_count', 0),
                    'edge_count': major_network.get('edge_count', 0),
                    'avg_degree': major_network.get('avg_degree', 0),
                    'max_degree': major_network.get('max_degree', 0),
                    'density': major_network.get('density', 0),
                    'alpha_index': major_network.get('alpha_index', 0),
                    'beta_index': major_network.get('beta_index', 0),
                    'gamma_index': major_network.get('gamma_index', 0),
                    'road_density': major_network.get('road_density', 0),
                    'avg_circuity': major_network.get('avg_circuity', 0)
                }
                
                f.write(f"| {city_jp} | {major_network.get('node_count', 0):,} | "
                       f"{major_network.get('edge_count', 0):,} | "
                       f"{major_network.get('avg_degree', 0):.2f} | "
                       f"{major_network.get('max_degree', 0)} | "
                       f"{major_network.get('density', 0):.6f} |\n")
            
            # Space Syntax 指標比較
            f.write("\n### Space Syntax 指標\n\n")
            f.write("| 都市 | α指数 | β指数 | γ指数 | 道路密度 (km/km²) | 平均迂回率 |\n")
            f.write("|------|-------|-------|-------|------------------|------------|\n")
            
            for location, results, analysis_type, execution_time in successful_analyses:
                city_jp = CITY_NAMES_JP.get(location, location)
                basic_analysis = results.get('basic_analysis', {})
                major_network = basic_analysis.get('major_network', {})
                
                f.write(f"| {city_jp} | "
                       f"{major_network.get('alpha_index', 0):.2f} | "
                       f"{major_network.get('beta_index', 0):.2f} | "
                       f"{major_network.get('gamma_index', 0):.2f} | "
                       f"{major_network.get('road_density', 0):.2f} | "
                       f"{major_network.get('avg_circuity', 0):.2f} |\n")
            
            # 統合評価比較（拡張分析がある場合）
            has_integrated_evaluation = any(
                results.get('integrated_evaluation') and not results.get('integrated_evaluation', {}).get('error')
                for _, results, _, _ in successful_analyses
            )
            
            if has_integrated_evaluation:
                f.write("\n## 🎯 統合評価比較\n\n")
                f.write("| 都市 | 回遊性 | アクセス性 | 効率性 | 総合スコア | 評価レベル |\n")
                f.write("|------|--------|------------|--------|------------|------------|\n")
                
                for location, results, analysis_type, execution_time in successful_analyses:
                    city_jp = CITY_NAMES_JP.get(location, location)
                    integrated_evaluation = results.get('integrated_evaluation', {})
                    
                    if integrated_evaluation and not integrated_evaluation.get('error'):
                        f.write(f"| {city_jp} | "
                               f"{integrated_evaluation.get('connectivity_score', 0):.1f} | "
                               f"{integrated_evaluation.get('accessibility_score', 0):.1f} | "
                               f"{integrated_evaluation.get('efficiency_score', 0):.1f} | "
                               f"{integrated_evaluation.get('overall_score', 0):.1f} | "
                               f"{integrated_evaluation.get('evaluation_level', '不明')} |\n")
                    else:
                        f.write(f"| {city_jp} | - | - | - | - | 評価不可 |\n")
            
            # 比較分析と考察
            f.write("\n## 🔍 比較分析と考察\n\n")
            
            # ランキング分析
            f.write("### 主要指標ランキング\n\n")
            
            # ネットワーク規模ランキング
            if network_data:
                size_ranking = sorted(network_data.items(), 
                                    key=lambda x: x[1]['node_count'], reverse=True)
                f.write("#### ネットワーク規模 (ノード数)\n")
                for i, (city, data) in enumerate(size_ranking, 1):
                    f.write(f"{i}. **{city}**: {data['node_count']:,}ノード\n")
                
                # α指数ランキング（循環性）
                alpha_ranking = sorted(network_data.items(), 
                                     key=lambda x: x[1]['alpha_index'], reverse=True)
                f.write("\n#### 循環性 (α指数)\n")
                for i, (city, data) in enumerate(alpha_ranking, 1):
                    f.write(f"{i}. **{city}**: {data['alpha_index']:.2f}\n")
                
                # 道路密度ランキング
                density_ranking = sorted(network_data.items(), 
                                       key=lambda x: x[1]['road_density'], reverse=True)
                f.write("\n#### 道路密度\n")
                for i, (city, data) in enumerate(density_ranking, 1):
                    f.write(f"{i}. **{city}**: {data['road_density']:.2f} km/km²\n")
                
                # 効率性ランキング（迂回率の逆順）
                efficiency_ranking = sorted(network_data.items(), 
                                          key=lambda x: x[1]['avg_circuity'])
                f.write("\n#### 道路効率性 (低迂回率順)\n")
                for i, (city, data) in enumerate(efficiency_ranking, 1):
                    f.write(f"{i}. **{city}**: {data['avg_circuity']:.2f}\n")
            
            # 都市特性分析
            f.write("\n### 都市特性分析\n\n")
            
            for location, results, analysis_type, execution_time in successful_analyses:
                city_jp = CITY_NAMES_JP.get(location, location)
                basic_analysis = results.get('basic_analysis', {})
                major_network = basic_analysis.get('major_network', {})
                
                f.write(f"#### {city_jp}\n\n")
                
                # 基本特性
                node_count = major_network.get('node_count', 0)
                edge_count = major_network.get('edge_count', 0)
                alpha_index = major_network.get('alpha_index', 0)
                gamma_index = major_network.get('gamma_index', 0)
                road_density = major_network.get('road_density', 0)
                avg_circuity = major_network.get('avg_circuity', 0)
                
                f.write(f"**基本特性**:\n")
                f.write(f"- ネットワーク規模: {node_count:,}ノード, {edge_count:,}エッジ\n")
                f.write(f"- 道路密度: {road_density:.2f} km/km²\n")
                f.write(f"- 平均迂回率: {avg_circuity:.2f}\n\n")
                
                # 都市構造の特徴分析
                f.write(f"**都市構造の特徴**:\n")
                
                # α指数による循環性評価
                if alpha_index > 30:
                    circulation = "高い循環性 - 多くの環状路がある複雑な道路網"
                elif alpha_index > 15:
                    circulation = "中程度の循環性 - バランスの取れた道路構造"
                else:
                    circulation = "低い循環性 - 樹状構造が主体の道路網"
                f.write(f"- 循環性 (α={alpha_index:.1f}): {circulation}\n")
                
                # γ指数による接続性評価
                if gamma_index > 0.7:
                    connectivity = "高い接続性 - 密な道路ネットワーク"
                elif gamma_index > 0.5:
                    connectivity = "中程度の接続性 - 標準的な道路密度"
                else:
                    connectivity = "低い接続性 - 疎な道路ネットワーク"
                f.write(f"- 接続性 (γ={gamma_index:.2f}): {connectivity}\n")
                
                # 道路密度による都市化度評価
                if road_density > 15:
                    urbanization = "高密度都市部 - 高度に都市化された地域"
                elif road_density > 8:
                    urbanization = "中密度市街地 - 適度に発達した市街地"
                else:
                    urbanization = "低密度地域 - 郊外または地方都市特性"
                f.write(f"- 都市化度: {urbanization}\n")
                
                # 迂回率による効率性評価
                if avg_circuity < 1.1:
                    efficiency = "非常に効率的 - 直線的な道路構造"
                elif avg_circuity < 1.3:
                    efficiency = "効率的 - 比較的直線的な移動が可能"
                elif avg_circuity < 1.5:
                    efficiency = "標準的効率性 - 一般的な迂回レベル"
                else:
                    efficiency = "非効率的 - 迂回が多い複雑な道路構造"
                f.write(f"- 移動効率性: {efficiency}\n\n")
            
            # 都市間比較による総合考察
            f.write("\n### 総合考察\n\n")
            
            if network_data:
                # 最大・最小値の都市を特定
                max_alpha_city = max(network_data.items(), key=lambda x: x[1]['alpha_index'])
                max_density_city = max(network_data.items(), key=lambda x: x[1]['road_density'])
                min_circuity_city = min(network_data.items(), key=lambda x: x[1]['avg_circuity'])
                max_size_city = max(network_data.items(), key=lambda x: x[1]['node_count'])
                
                f.write(f"**長野県主要都市の Space Syntax 分析から得られた知見**:\n\n")
                
                f.write(f"1. **最も複雑な道路構造**: {max_alpha_city[0]} (α指数: {max_alpha_city[1]['alpha_index']:.2f})\n")
                f.write(f"   - 環状路や代替ルートが豊富で、交通の分散効果が期待できる\n\n")
                
                f.write(f"2. **最も高密度な道路網**: {max_density_city[0]} (道路密度: {max_density_city[1]['road_density']:.2f} km/km²)\n")
                f.write(f"   - 高度に都市化されており、アクセス性に優れている\n\n")
                
                f.write(f"3. **最も効率的な移動**: {min_circuity_city[0]} (平均迂回率: {min_circuity_city[1]['avg_circuity']:.2f})\n")
                f.write(f"   - 直線的な移動が可能で、時間効率の良い交通が期待できる\n\n")
                
                f.write(f"4. **最大規模のネットワーク**: {max_size_city[0]} ({max_size_city[1]['node_count']:,}ノード)\n")
                f.write(f"   - 広域的な交通結節点としての機能を持つ\n\n")
                
                # 地域特性に基づく考察
                f.write(f"**地域特性に基づく分析**:\n\n")
                f.write(f"長野県の各都市は、それぞれ異なる地理的・歴史的背景を持っており、")
                f.write(f"道路ネットワークの構造にもその特徴が反映されています:\n\n")
                
                for city_jp in analyzed_cities:
                    if city_jp == "松本市":
                        f.write(f"- **松本市**: 松本盆地の中心都市として、比較的計画的な道路配置が")
                        f.write(f"見られる可能性があります。城下町としての歴史的街区と、")
                        f.write(f"現代的な都市計画が混在した構造が特徴的です。\n\n")
                    elif city_jp == "長野市":
                        f.write(f"- **長野市**: 善光寺を中心とした放射状の道路構造と、")
                        f.write(f"県庁所在地としての広域的な交通結節機能を併せ持つと")
                        f.write(f"考えられます。歴史的街区と新市街地の混在が特徴です。\n\n")
                    elif city_jp == "上田市":
                        f.write(f"- **上田市**: 上田盆地の地形制約と、真田氏の城下町として")
                        f.write(f"の歴史的な街区構造が、現在の道路ネットワークに影響を")
                        f.write(f"与えている可能性があります。\n\n")
            
            # 今後の活用方針
            f.write(f"### 今後の活用方針\n\n")
            f.write(f"**都市計画への応用**:\n")
            f.write(f"- 循環性の低い都市では、環状道路や代替ルートの整備を検討\n")
            f.write(f"- 迂回率の高い都市では、直線的なアクセス路の改善を検討\n")
            f.write(f"- 道路密度の格差を考慮した、バランスの取れた交通インフラ整備\n\n")
            
            f.write(f"**交通政策への示唆**:\n")
            f.write(f"- 各都市の道路構造特性に応じた交通流制御\n")
            f.write(f"- 公共交通システムの最適配置計画\n")
            f.write(f"- 災害時の代替ルート確保戦略\n\n")
            
            # メタデータ
            f.write(f"---\n\n")
            f.write(f"## 📋 分析メタデータ\n\n")
            f.write(f"- **分析手法**: Space Syntax Analysis\n")
            f.write(f"- **使用データ**: OpenStreetMap\n")
            f.write(f"- **分析ツール**: OSMnx, NetworkX\n")
            f.write(f"- **分析日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"- **レポート形式**: Markdown\n")
            f.write(f"- **文字エンコーディング**: UTF-8\n\n")
            
            # 免責事項
            f.write(f"## ⚠️ 免責事項\n\n")
            f.write(f"本分析結果は OpenStreetMap データに基づく理論的分析であり、")
            f.write(f"実際の交通流や都市機能を完全に反映するものではありません。")
            f.write(f"都市計画や政策決定の際には、本分析結果を参考資料の一つとして")
            f.write(f"活用し、他の調査・データと総合的に検討することを推奨します。\n")
        
        print(f"📋 比較分析レポート生成完了: {report_file.name}")
        
        # 追加: CSVサマリーも生成
        generate_csv_summary(successful_analyses, output_dir, timestamp)
        
    except Exception as e:
        logger.error(f"レポート生成エラー: {e}")
        print(f"❌ レポート生成でエラーが発生しました: {e}")


def generate_csv_summary(successful_analyses, output_dir, timestamp):
    """CSV形式のサマリーデータを生成"""
    try:
        csv_file = output_dir / f"analysis_summary_{timestamp}.csv"
        
        # データフレーム用のデータを準備
        summary_data = []
        
        for location, results, analysis_type, execution_time in successful_analyses:
            city_jp = CITY_NAMES_JP.get(location, location)
            basic_analysis = results.get('basic_analysis', {})
            major_network = basic_analysis.get('major_network', {})
            integrated_evaluation = results.get('integrated_evaluation', {})
            
            row_data = {
                '都市名': city_jp,
                '英語名': location,
                '分析タイプ': analysis_type,
                '実行時間_秒': execution_time,
                'ノード数': major_network.get('node_count', 0),
                'エッジ数': major_network.get('edge_count', 0),
                '平均次数': major_network.get('avg_degree', 0),
                '最大次数': major_network.get('max_degree', 0),
                '密度': major_network.get('density', 0),
                'α指数': major_network.get('alpha_index', 0),
                'β指数': major_network.get('beta_index', 0),
                'γ指数': major_network.get('gamma_index', 0),
                '道路密度_km_per_km2': major_network.get('road_density', 0),
                '平均迂回率': major_network.get('avg_circuity', 0),
                '連結成分数': major_network.get('num_components', 0)
            }
            
            # 統合評価データ（拡張分析の場合）
            if integrated_evaluation and not integrated_evaluation.get('error'):
                row_data.update({
                    '回遊性スコア': integrated_evaluation.get('connectivity_score', 0),
                    'アクセス性スコア': integrated_evaluation.get('accessibility_score', 0),
                    '効率性スコア': integrated_evaluation.get('efficiency_score', 0),
                    '総合スコア': integrated_evaluation.get('overall_score', 0),
                    '評価レベル': integrated_evaluation.get('evaluation_level', '')
                })
            else:
                row_data.update({
                    '回遊性スコア': None,
                    'アクセス性スコア': None,
                    '効率性スコア': None,
                    '総合スコア': None,
                    '評価レベル': ''
                })
            
            summary_data.append(row_data)
        
        # CSVファイルに書き込み
        import pandas as pd
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')  # Excel対応のためBOM付きUTF-8
        
        print(f"📊 CSVサマリー生成完了: {csv_file.name}")
        
    except Exception as e:
        logger.warning(f"CSVサマリー生成エラー: {e}")


def demo_axial_analysis_only():
    """軸線分析単体デモ"""
    print("\n🔄 軸線分析単体デモ実行中...")
    
    try:
        from space_syntax_analyzer.core.axial import AxialAnalyzer
        
        axial_analyzer = AxialAnalyzer()
        print("✅ 軸線分析アナライザー初期化成功")
        
        # ダミーネットワークでの分析
        import networkx as nx
        dummy_network = nx.grid_2d_graph(5, 5)
        
        results = axial_analyzer.calculate_axial_summary(dummy_network)
        print(f"✅ 軸線分析完了: {results.get('network_metrics', {})}")
        
    except Exception as e:
        logger.error(f"軸線分析単体デモエラー: {e}")
        print("❌ 軸線分析単体デモも失敗しました")


def demo_basic_analysis_fallback():
    """基本分析フォールバック"""
    print("\n🔄 基本分析フォールバック実行中...")
    
    try:
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        analyzer = SpaceSyntaxAnalyzer()
        
        # フォールバック時も長野県都市を使用
        fallback_location = "Matsumoto, Nagano, Japan"  # 松本市をフォールバック地点に
        print(f"📍 フォールバック分析対象: 松本市")
        
        results = analyzer.analyze_place(fallback_location)
        
        if not results.get('error', False):
            print("✅ 基本分析成功")
            display_basic_results(results)
        else:
            print(f"❌ 基本分析エラー: {results.get('error_message', '不明')}")
        
    except Exception as e:
        logger.error(f"基本分析フォールバックエラー: {e}")
        print("❌ 基本分析フォールバックも失敗しました")


def main():
    """メイン関数"""
    print("🌟 最適化版拡張 Space Syntax Analyzer デモンストレーション")
    print("="*80)
    
    # 拡張機能依存関係チェック
    print("🔍 拡張機能依存関係チェック中...")
    
    # 必須ライブラリのチェック
    required_libs = ["osmnx", "networkx", "pandas", "matplotlib", "numpy", "scipy", "shapely"]
    optional_libs = ["geopandas", "scikit-learn", "plotly", "folium"]
    
    missing_required = []
    missing_optional = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"   ✅ {lib}")
        except ImportError:
            print(f"   ❌ {lib}")
            missing_required.append(lib)
    
    for lib in optional_libs:
        try:
            __import__(lib)
            print(f"   ✅ {lib} (オプション)")
        except ImportError:
            print(f"   ⚠️ {lib} (オプション・未インストール)")
            missing_optional.append(lib)
    
    if missing_required:
        print(f"\n❌ 必須ライブラリが不足: {', '.join(missing_required)}")
        print("インストール: uv add " + " ".join(missing_required))
        return
    
    if missing_optional:
        print(f"\n⚠️ オプション機能制限: {', '.join(missing_optional)}")
        print("フル機能使用には: uv add " + " ".join(missing_optional))
    
    print("✅ 最適化拡張機能の実行が可能です")
    
    # 日本語フォント状況の表示
    if JAPANESE_FONT_AVAILABLE:
        print("✅ 日本語チャート生成が可能です")
    else:
        print("⚠️ 日本語フォント未対応 - 英語チャートで代替します")
    
    # 出力ディレクトリの作成
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir.absolute()}")
    
    # デモ選択メニュー
    print("\n📋 利用可能なデモ:")
    print("   1: 最適化包括分析デモ（推奨）")
    print("   2: 軸線分析単体デモ")
    print("   3: 基本機能デモ")
    print("   a: 全デモ実行")
    
    try:
        choice = input("\n選択してください (1-3, a, q=終了): ").strip().lower()
        
        if choice == 'q':
            print("👋 デモを終了します")
            return
        elif choice == '1':
            demo_enhanced_comprehensive_analysis()
        elif choice == '2':
            demo_axial_analysis_only()
        elif choice == '3':
            demo_basic_analysis_fallback()
        elif choice == 'a':
            print("📋 全デモ実行:")
            demo_enhanced_comprehensive_analysis()
            demo_axial_analysis_only()
            demo_basic_analysis_fallback()
        else:
            print("❌ 無効な選択です")
            return
        
    except KeyboardInterrupt:
        print("\n👋 ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
        print(f"❌ 予期しないエラー: {e}")
    
    print(f"\n📁 結果ファイルは {output_dir.absolute()} に保存されています")
    print(f"💡 問題が発生した場合は、ログを確認するか座標指定での分析をお試しください")


if __name__ == "__main__":
    main()