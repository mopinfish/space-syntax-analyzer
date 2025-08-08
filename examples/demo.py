# demo_fixed.py
"""
修正版 Space Syntax Analyzer デモスクリプト（最終版）

既存のプロジェクト構造に合わせて、堅牢な分析機能を提供
"""

import logging
import os
import sys
from pathlib import Path
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_robust_basic_analysis():
    """堅牢な基本分析のデモンストレーション"""
    print("="*50)
    print("🚀 堅牢な基本分析デモンストレーション")
    print("="*50)
    
    try:
        # 既存のspace_syntax_analyzerモジュールを使用
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        # アナライザーの初期化
        analyzer = SpaceSyntaxAnalyzer()
        
        # 複数の地名で試行（成功するまで）
        test_locations = [
            "渋谷, 東京",
            "Shibuya, Tokyo", 
            "Tokyo, Japan",
            "新宿, 東京",
            "Shinjuku, Tokyo"
        ]
        
        successful_analysis = None
        for location in test_locations:
            print(f"\n📍 分析試行: {location}")
            
            try:
                # 拡張されたanalyze_placeメソッドを使用
                results = analyzer.analyze_place(location, analysis_types=["basic", "connectivity"])
                
                # エラーチェック
                if results.get('error', False):
                    print(f"❌ 分析エラー: {results.get('error_message', '不明')}")
                    print("💡 提案された解決策:")
                    for suggestion in results.get('suggestions', []):
                        print(f"   • {suggestion}")
                    continue
                
                # 成功
                successful_analysis = (location, results)
                break
                
            except Exception as e:
                print(f"❌ 例外発生: {e}")
                continue
        
        if successful_analysis:
            location, results = successful_analysis
            print(f"\n✅ 分析成功: {location}")
            
            # 結果の表示
            report = analyzer.generate_report(results, f"{location} 分析結果")
            print(report)
            
            # 基本統計の表示
            major_data = results.get('major_network', {})
            if major_data:
                print(f"\n📊 主要道路ネットワーク統計:")
                print(f"   ノード数: {major_data.get('node_count', 0):,}")
                print(f"   エッジ数: {major_data.get('edge_count', 0):,}")
                print(f"   平均次数: {major_data.get('avg_degree', 0):.2f}")
                print(f"   密度: {major_data.get('density', 0):.4f}")
                print(f"   連結性: {'✓' if major_data.get('is_connected', False) else '✗'}")
            
            # 統合評価の表示
            integration = results.get('integration_summary', {})
            if integration and 'overall_integration_score' in integration:
                print(f"\n🎯 統合評価:")
                print(f"   総合スコア: {integration['overall_integration_score']:.1f}/100")
                print(f"   評価レベル: {integration.get('integration_level', '不明')}")
            
            # 結果のエクスポート試行
            try:
                output_dir = Path("demo_output")
                output_dir.mkdir(exist_ok=True)
                
                export_success = analyzer.export_results(
                    results, 
                    str(output_dir / f"analysis_{location.replace(',', '_').replace(' ', '_')}.csv")
                )
                
                if export_success:
                    print(f"✅ 分析結果を {output_dir} に保存")
                else:
                    print("⚠️  エクスポートに失敗しましたが、分析は成功しました")
                
            except Exception as e:
                print(f"⚠️  エクスポートエラー: {e}")
            
            # 可視化試行
            try:
                print("\n📊 可視化を試行中...")
                # ネットワーク取得
                major_net, full_net = analyzer.get_network(location, "both")
                
                if major_net or full_net:
                    vis_success = analyzer.visualize(
                        major_net, full_net, results,
                        str(output_dir / f"visualization_{location.replace(',', '_').replace(' ', '_')}.png")
                    )
                    if vis_success:
                        print(f"✅ 可視化結果を {output_dir} に保存")
                    else:
                        print("⚠️  可視化に失敗しましたが、分析は成功しました")
                else:
                    print("⚠️  ネットワーク取得に失敗したため可視化をスキップ")
                
            except Exception as e:
                print(f"⚠️  可視化エラー: {e}")
        else:
            print("❌ すべての地名で分析に失敗しました")
            print("💡 座標指定での分析をお試しください")
            
            # 座標での分析例
            demo_coordinate_analysis()
        
    except ImportError as e:
        print(f"❌ モジュールインポートエラー: {e}")
        print("💡 必要なモジュールがインストールされているか確認してください")
        print("   pip install osmnx networkx pandas matplotlib numpy")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")


def demo_coordinate_analysis():
    """座標指定分析のデモンストレーション"""
    print("\n" + "="*50)
    print("📍 座標指定分析デモンストレーション")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        analyzer = SpaceSyntaxAnalyzer()
        
        # 東京の主要地点の座標
        test_coordinates = [
            (35.6580, 139.7016, "東京駅周辺"),  # 東京駅
            (35.6762, 139.6503, "新宿周辺"),    # 新宿
            (35.6596, 139.7006, "銀座周辺"),    # 銀座
        ]
        
        for lat, lon, description in test_coordinates:
            print(f"\n📍 座標分析: {description} ({lat:.4f}, {lon:.4f})")
            
            try:
                # analyze_pointメソッドが実装されていれば使用
                if hasattr(analyzer, 'analyze_point'):
                    results = analyzer.analyze_point(lat, lon, radius=800)
                else:
                    # フォールバック: analyze_placeで座標文字列を使用
                    coord_string = f"{lat}, {lon}"
                    results = analyzer.analyze_place(coord_string)
                
                if results.get('error', False):
                    print(f"❌ 分析エラー: {results.get('error_message', '不明')}")
                    continue
                
                print(f"✅ 分析成功")
                
                # 簡易レポート表示
                if 'major_network' in results:
                    major_stats = results['major_network']
                    print(f"   ノード数: {major_stats.get('node_count', 0)}")
                    print(f"   エッジ数: {major_stats.get('edge_count', 0)}")
                    print(f"   平均次数: {major_stats.get('avg_degree', 0):.2f}")
                
                integration = results.get('integration_summary', {})
                if integration and 'overall_integration_score' in integration:
                    print(f"   総合スコア: {integration['overall_integration_score']:.1f}/100")
                    print(f"   評価: {integration.get('integration_level', '不明')}")
                
                # 最初の成功例のみ詳細表示
                if description == "東京駅周辺":
                    report = analyzer.generate_report(results, description)
                    print(f"\n詳細レポート:\n{report}")
                    
                    # エクスポート
                    output_dir = Path("demo_output")
                    output_dir.mkdir(exist_ok=True)
                    analyzer.export_results(results, str(output_dir / f"coordinate_analysis_{description}.csv"))
                
                break  # 最初の成功で終了
                
            except Exception as e:
                print(f"❌ 分析エラー: {e}")
                continue
    
    except Exception as e:
        print(f"❌ 座標分析デモエラー: {e}")


def demo_error_handling():
    """エラーハンドリング機能のデモンストレーション"""
    print("\n" + "="*50)
    print("🛡️ エラーハンドリング機能デモ")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        analyzer = SpaceSyntaxAnalyzer()
        
        # 意図的に問題のある入力でテスト
        problematic_inputs = [
            "存在しない地名12345",
            "Invalid Location XYZ",
            "あいうえお",
        ]
        
        for problematic_input in problematic_inputs:
            print(f"\n🧪 問題のある入力をテスト: '{problematic_input}'")
            
            try:
                results = analyzer.analyze_place(problematic_input)
                
                if results.get('error', False):
                    print(f"✅ エラーが適切に処理されました")
                    print(f"   エラータイプ: {results.get('error_type', '不明')}")
                    print(f"   エラー内容: {results.get('error_message', '不明')}")
                    print(f"   提案された解決策:")
                    for suggestion in results.get('suggestions', []):
                        print(f"     • {suggestion}")
                else:
                    print(f"⚠️  予期せず成功しました（これは正常な場合もあります）")
                    
            except Exception as e:
                print(f"❌ 予期しない例外: {e}")
    
    except Exception as e:
        print(f"❌ エラーハンドリングデモエラー: {e}")


def demo_performance_comparison():
    """パフォーマンス比較デモ"""
    print("\n" + "="*50)
    print("⚡ パフォーマンス比較デモ")
    print("="*50)
    
    try:
        from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
        
        analyzer = SpaceSyntaxAnalyzer()
        
        # 座標ベースで異なるサイズの分析を実行
        test_scenarios = [
            (35.6580, 139.7016, 500, "小規模(500m)"),
            (35.6580, 139.7016, 1000, "中規模(1000m)"),
            (35.6580, 139.7016, 1500, "大規模(1500m)"),
        ]
        
        results_summary = []
        
        for lat, lon, radius, description in test_scenarios:
            print(f"\n⏱️  {description} パフォーマンステスト")
            
            try:
                start_time = time.time()
                
                # analyze_pointが利用可能かチェック
                if hasattr(analyzer, 'analyze_point'):
                    results = analyzer.analyze_point(lat, lon, radius=radius)
                else:
                    # NetworkManagerを直接使用
                    network = analyzer.network_manager.get_network_from_point((lat, lon), radius)
                    if network:
                        results = analyzer._analyze_network_safe(network, f"テスト({radius}m)", ["basic"])
                        results['location'] = f"({lat}, {lon})"
                    else:
                        raise Exception("ネットワーク取得失敗")
                
                end_time = time.time()
                
                if not results.get('error', False):
                    node_count = 0
                    edge_count = 0
                    
                    if 'major_network' in results:
                        major_stats = results['major_network']
                        node_count = major_stats.get('node_count', 0)
                        edge_count = major_stats.get('edge_count', 0)
                    elif 'node_count' in results:
                        node_count = results.get('node_count', 0)
                        edge_count = results.get('edge_count', 0)
                    
                    execution_time = end_time - start_time
                    
                    print(f"   ✅ 完了時間: {execution_time:.1f}秒")
                    print(f"   ネットワークサイズ: {node_count}ノード, {edge_count}エッジ")
                    print(f"   処理速度: {node_count/execution_time:.1f}ノード/秒")
                    
                    results_summary.append({
                        'scenario': description,
                        'nodes': node_count,
                        'edges': edge_count,
                        'time': execution_time,
                        'speed': node_count/execution_time if execution_time > 0 else 0
                    })
                else:
                    print(f"   ❌ 分析失敗: {results.get('error_message', '不明')}")
                    
            except Exception as e:
                print(f"   ❌ エラー: {e}")
        
        # パフォーマンス結果サマリー
        if results_summary:
            print(f"\n📊 パフォーマンス結果サマリー:")
            print(f"{'シナリオ':<15} {'ノード数':<10} {'処理時間':<10} {'処理速度':<15}")
            print("-" * 60)
            
            for result in results_summary:
                print(f"{result['scenario']:<15} "
                      f"{result['nodes']:<10} "
                      f"{result['time']:<10.1f} "
                      f"{result['speed']:<15.1f}")
    
    except Exception as e:
        print(f"❌ パフォーマンス比較デモエラー: {e}")


def check_dependencies():
    """依存関係チェック"""
    print("🔍 依存関係チェック中...")
    
    required_packages = [
        'osmnx',
        'networkx', 
        'pandas',
        'matplotlib',
        'numpy',
        'geopandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (未インストール)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  不足パッケージ: {', '.join(missing_packages)}")
        print(f"インストールコマンド: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ すべての依存関係が満たされています")
        return True


def main():
    """メインデモ関数"""
    print("🌟 修正版 Space Syntax Analyzer デモンストレーション")
    print("="*70)
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n❌ 依存関係の問題により、デモを続行できません")
        return
    
    # 出力ディレクトリの作成
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir.absolute()}")
    
    # 自動実行モード判定
    auto_mode = os.getenv('DEMO_AUTO_MODE', 'false').lower() == 'true'
    
    if auto_mode:
        print(f"\n🤖 自動実行モード: 堅牢な基本分析を実行")
        demo_robust_basic_analysis()
    else:
        # インタラクティブモード
        demos = [
            ("1", "堅牢な基本分析デモ", demo_robust_basic_analysis),
            ("2", "座標指定分析デモ", demo_coordinate_analysis),
            ("3", "エラーハンドリングデモ", demo_error_handling),
            ("4", "パフォーマンス比較デモ", demo_performance_comparison),
            ("a", "全デモ実行", None),
        ]
        
        print(f"\n📋 利用可能なデモ:")
        for code, name, _ in demos:
            print(f"   {code}: {name}")
        
        choice = input(f"\n選択してください (1-4, a, q=終了): ").strip().lower()
        
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
                    time.sleep(1)  # 少し間隔を開ける
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
    print(f"💡 問題が発生した場合は、ログを確認するか座標指定での分析をお試しください")


if __name__ == "__main__":
    main()