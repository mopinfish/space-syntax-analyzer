#!/usr/bin/env python3
"""
Space Syntax解析システム メインモジュール
パス: main.py

OpenStreetMapデータを用いたSpace Syntax解析の実行制御
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.config_manager import ConfigManager
from src.osm_data_loader import OSMDataLoader
from src.graph_builder import GraphBuilder
from src.space_syntax_analyzer import SpaceSyntaxAnalyzer
from src.visualization import Visualizer
from src.report_generator import ReportGenerator
from src.utils import setup_logging, validate_coordinates


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description='Space Syntax解析システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py --place "Shibuya, Tokyo, Japan"
  python main.py --bbox 35.6580,139.6956,35.6650,139.7056
  python main.py --config custom_config.json
        """
    )
    
    # 地域指定方式
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        '--place', 
        type=str,
        help='解析対象地域名（例: "Shibuya, Tokyo, Japan"）'
    )
    location_group.add_argument(
        '--bbox',
        type=str,
        help='境界座標（south,west,north,east形式）'
    )
    location_group.add_argument(
        '--admin',
        type=str,
        help='行政区域名'
    )
    
    # 設定ファイル
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.json',
        help='設定ファイルパス（デフォルト: config/default_config.json）'
    )
    
    # 出力ディレクトリ
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='出力ディレクトリ（デフォルト: output）'
    )
    
    # 解析タイプ
    parser.add_argument(
        '--analysis-type',
        choices=['axial', 'segment', 'both'],
        default='both',
        help='解析タイプ（デフォルト: both）'
    )
    
    # ログレベル
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル（デフォルト: INFO）'
    )
    
    # 静音モード
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静音モード（進捗表示を抑制）'
    )
    
    return parser.parse_args()


def validate_bbox(bbox_str: str) -> tuple:
    """境界座標の検証"""
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError("境界座標は4つの値が必要です")
        
        south, west, north, east = coords
        
        if not validate_coordinates(south, west) or not validate_coordinates(north, east):
            raise ValueError("座標値が有効範囲外です")
        
        if south >= north or west >= east:
            raise ValueError("境界座標の順序が正しくありません")
        
        return (south, west, north, east)
    
    except ValueError as e:
        raise ValueError(f"境界座標の形式が正しくありません: {e}")


def main():
    """メイン処理"""
    # 引数解析
    args = parse_arguments()
    
    # ログ設定
    setup_logging(level=args.log_level, quiet=args.quiet)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Space Syntax解析システム開始 ===")
    start_time = datetime.now()
    
    try:
        # 出力ディレクトリ作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定ファイル読み込み
        logger.info(f"設定ファイル読み込み: {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        
        # 地域パラメータ設定
        location_params = {}
        
        if args.place:
            location_params['place'] = args.place
            logger.info(f"解析対象地域: {args.place}")
            
        elif args.bbox:
            bbox = validate_bbox(args.bbox)
            location_params['bbox'] = bbox
            logger.info(f"解析対象境界: {bbox}")
            
        elif args.admin:
            location_params['admin'] = args.admin
            logger.info(f"解析対象行政区域: {args.admin}")
        
        # データ取得
        logger.info("OpenStreetMapデータ取得開始")
        data_loader = OSMDataLoader(config['osm_settings'])
        
        if 'place' in location_params:
            graph = data_loader.load_by_place(location_params['place'])
        elif 'bbox' in location_params:
            graph = data_loader.load_by_bbox(location_params['bbox'])
        elif 'admin' in location_params:
            graph = data_loader.load_by_admin(location_params['admin'])
        
        logger.info(f"グラフ構築完了: {len(graph.nodes)}ノード, {len(graph.edges)}エッジ")
        
        # グラフ変換
        logger.info("グラフ構造構築開始")
        graph_builder = GraphBuilder(config['graph_settings'])
        
        results = {}
        
        # Axial Map解析
        if args.analysis_type in ['axial', 'both']:
            logger.info("Axial Map解析開始")
            axial_graph = graph_builder.build_axial_map(graph)
            
            analyzer = SpaceSyntaxAnalyzer(config['analysis_settings'])
            axial_results = analyzer.analyze_axial_map(axial_graph)
            
            # Space Syntax理論準拠の正規化適用
            axial_results = analyzer.apply_normalization(axial_results)
            
            # Four-Pointed Star Model計算（正規化済みデータ使用）
            if 'NACH_' in str(axial_results['metrics'].keys()) and 'NAIN_' in str(axial_results['metrics'].keys()):
                nach_metric = next((v for k, v in axial_results['metrics'].items() if k.startswith('NACH_')), None)
                nain_metric = next((v for k, v in axial_results['metrics'].items() if k.startswith('NAIN_')), None)
                
                if nach_metric is not None and nain_metric is not None:
                    four_star = analyzer.calculate_four_pointed_star(nach_metric, nain_metric)
                    axial_results['four_pointed_star'] = four_star
            
            results['axial'] = axial_results
            logger.info("Axial Map解析完了")
        
        # Segment Map解析
        if args.analysis_type in ['segment', 'both']:
            logger.info("Segment Map解析開始")
            segment_graph = graph_builder.build_segment_map(graph)
            
            analyzer = SpaceSyntaxAnalyzer(config['analysis_settings'])
            segment_results = analyzer.analyze_segment_map(segment_graph)
            
            # Space Syntax理論準拠の正規化適用
            segment_results = analyzer.apply_normalization(segment_results)
            
            # Four-Pointed Star Model計算
            if 'NACH_' in str(segment_results['metrics'].keys()) and 'NAIN_' in str(segment_results['metrics'].keys()):
                nach_metric = next((v for k, v in segment_results['metrics'].items() if k.startswith('NACH_')), None)
                nain_metric = next((v for k, v in segment_results['metrics'].items() if k.startswith('NAIN_')), None)
                
                if nach_metric is not None and nain_metric is not None:
                    four_star = analyzer.calculate_four_pointed_star(nach_metric, nain_metric)
                    segment_results['four_pointed_star'] = four_star
            
            results['segment'] = segment_results
            logger.info("Segment Map解析完了")
        
        # 可視化
        logger.info("可視化処理開始")
        visualizer = Visualizer(config['visualization_settings'])
        
        for analysis_type, data in results.items():
            viz_output_dir = output_dir / analysis_type
            viz_output_dir.mkdir(exist_ok=True)
            
            # 地図可視化
            visualizer.create_map_visualizations(
                data['graph'], 
                data['metrics'], 
                viz_output_dir
            )
            
            # 統計グラフ
            visualizer.create_statistical_plots(
                data['metrics'], 
                viz_output_dir
            )
        
        logger.info("可視化処理完了")
        
        # レポート生成
        logger.info("レポート生成開始")
        report_generator = ReportGenerator(config['report_settings'])
        
        report_path = output_dir / "space_syntax_report.pdf"
        report_generator.generate_comprehensive_report(
            results,
            location_params,
            report_path
        )
        
        logger.info(f"レポート生成完了: {report_path}")
        
        # 実行時間計算
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info(f"=== 解析完了 ===")
        logger.info(f"実行時間: {execution_time}")
        logger.info(f"出力ディレクトリ: {output_dir.absolute()}")
        
        # 結果サマリー出力
        if not args.quiet:
            print("\n" + "="*50)
            print("Space Syntax解析結果サマリー")
            print("="*50)
            
            for analysis_type, data in results.items():
                print(f"\n{analysis_type.upper()} MAP解析:")
                metrics = data['metrics']
                
                if 'integration' in metrics:
                    integration_stats = metrics['integration'].describe()
                    print(f"  Integration - 平均: {integration_stats['mean']:.4f}, "
                          f"標準偏差: {integration_stats['std']:.4f}")
                
                if 'connectivity' in metrics:
                    connectivity_stats = metrics['connectivity'].describe()
                    print(f"  Connectivity - 平均: {connectivity_stats['mean']:.4f}, "
                          f"標準偏差: {connectivity_stats['std']:.4f}")
            
            print(f"\n詳細な結果は {output_dir} ディレクトリをご確認ください。")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
