# examples/station_analysis_demo.py
"""
駅周辺道路ネットワーク分析デモ

設定ファイルから駅情報を読み込み、各駅周辺800m圏内の道路ネットワークを分析
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# 日本語フォント設定
try:
    import japanize_matplotlib
    JAPANESE_FONT_AVAILABLE = True
    print("✅ 日本語フォント設定完了")
except ImportError:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    JAPANESE_FONT_AVAILABLE = False
    print("⚠️ 日本語フォントが利用できません。英語フォントを使用します。")

import networkx as nx
import osmnx as ox
import pandas as pd

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 分析結果保存用
STATION_RESULTS = {}


class StationNetworkAnalyzer:
    """駅周辺ネットワーク分析クラス"""
    
    def __init__(self, config_path: str = "station_config.json"):
        """
        駅周辺ネットワーク分析器を初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config.get("output_directory", "station_analysis_output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Space Syntax Analyzer の初期化
        try:
            from space_syntax_analyzer.core.analyzer import SpaceSyntaxAnalyzer
            from space_syntax_analyzer.core.visualization import NetworkVisualizer
            
            self.analyzer = SpaceSyntaxAnalyzer()
            self.visualizer = NetworkVisualizer()
            print("✅ Space Syntax Analyzer 初期化完了")
        except ImportError as e:
            print(f"❌ Space Syntax Analyzer の初期化に失敗: {e}")
            sys.exit(1)
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            if not self.config_path.exists():
                print(f"❌ 設定ファイルが見つかりません: {self.config_path}")
                print("サンプル設定ファイルを生成しますか？ (y/n): ", end="")
                response = input().strip().lower()
                if response == 'y':
                    self._create_sample_config()
                    print(f"✅ サンプル設定ファイルを生成しました: {self.config_path}")
                    print("設定を編集してから再実行してください。")
                sys.exit(1)
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✅ 設定ファイル読み込み完了: {self.config_path}")
            return config
            
        except Exception as e:
            print(f"❌ 設定ファイル読み込みエラー: {e}")
            sys.exit(1)
    
    def _create_sample_config(self):
        """サンプル設定ファイルを生成"""
        sample_config = {
            "analysis_settings": {
                "radius_meters": 800,
                "network_type": "drive",
                "include_analysis": ["basic", "axial", "integration"],
                "save_graphml": True,
                "save_visualization": True,
                "background_map": True
            },
            "output_directory": "station_analysis_output",
            "stations": [
                {
                    "id": "shinjuku",
                    "name": "新宿駅",
                    "location": "Shinjuku Station, Tokyo, Japan",
                    "coordinates": null,
                    "graphml_path": null,
                    "description": "日本最大のターミナル駅"
                },
                {
                    "id": "shibuya", 
                    "name": "渋谷駅",
                    "location": "Shibuya Station, Tokyo, Japan",
                    "coordinates": [35.6580, 139.7016],
                    "graphml_path": null,
                    "description": "若者文化の中心地"
                },
                {
                    "id": "tokyo",
                    "name": "東京駅",
                    "location": "Tokyo Station, Tokyo, Japan",
                    "coordinates": null,
                    "graphml_path": "data/tokyo_station_network.graphml",
                    "description": "日本の鉄道網の中心"
                },
                {
                    "id": "matsumoto",
                    "name": "松本駅",
                    "location": "Matsumoto Station, Nagano, Japan",
                    "coordinates": [36.2408, 137.9677],
                    "graphml_path": null,
                    "description": "松本城の最寄り駅"
                }
            ]
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, ensure_ascii=False, indent=2)
    
    def analyze_all_stations(self) -> Dict[str, Any]:
        """全駅の分析を実行"""
        print("🚉 駅周辺ネットワーク分析を開始します")
        print("="*80)
        
        stations = self.config.get("stations", [])
        analysis_settings = self.config.get("analysis_settings", {})
        
        print(f"📋 分析対象駅数: {len(stations)}")
        print(f"📏 分析半径: {analysis_settings.get('radius_meters', 800)}m")
        print(f"🗺️ ネットワーク種別: {analysis_settings.get('network_type', 'drive')}")
        print()
        
        results = {}
        successful_analyses = []
        
        for i, station in enumerate(stations, 1):
            station_id = station.get("id", f"station_{i}")
            station_name = station.get("name", f"駅{i}")
            
            print(f"📍 [{i}/{len(stations)}] {station_name} ({station_id}) 分析中...")
            
            try:
                result = self._analyze_single_station(station, analysis_settings)
                
                if result and not result.get('error', False):
                    results[station_id] = result
                    successful_analyses.append((station_id, station_name, result))
                    print(f"✅ {station_name} 分析完了")
                    
                    # 可視化の保存
                    if analysis_settings.get('save_visualization', True):
                        self._save_station_visualizations(station, result)
                    
                else:
                    print(f"❌ {station_name} 分析失敗: {result.get('error_message', '不明')}")
                
            except Exception as e:
                print(f"❌ {station_name} 分析中にエラー: {e}")
                logger.error(f"駅分析エラー ({station_id}): {e}")
            
            print()
        
        # 結果の保存
        global STATION_RESULTS
        STATION_RESULTS = results
        
        # 比較分析の実行
        if len(successful_analyses) > 1:
            print("📊 駅間比較分析を実行中...")
            self._generate_comparative_analysis(successful_analyses)
        
        # 統合レポートの生成
        self._generate_station_report(successful_analyses)
        
        print(f"🎉 分析完了! 結果は {self.output_dir} に保存されました")
        return results
    
    def _analyze_single_station(self, station: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        """単一駅の分析"""
        station_id = station.get("id")
        station_name = station.get("name")
        
        # ネットワークの取得または読み込み
        network = self._get_or_load_network(station, settings)
        
        if network is None:
            return {
                "error": True,
                "error_message": "ネットワークの取得/読み込みに失敗"
            }
        
        # 基本分析の実行
        try:
            print(f"   🔍 基本分析実行中...")
            
            # NetworkXグラフから基本指標を直接計算
            basic_results = self._calculate_network_metrics(network)
            
            result = {
                "station_info": station,
                "network_data": {
                    "node_count": network.number_of_nodes(),
                    "edge_count": network.number_of_edges(),
                    "network_type": settings.get("network_type", "drive")
                },
                "basic_analysis": basic_results,
                "analysis_timestamp": datetime.now().isoformat(),
                "settings": settings
            }
            
            # 拡張分析（利用可能な場合）
            include_analysis = settings.get("include_analysis", [])
            
            if "axial" in include_analysis:
                print(f"   🔄 軸線分析実行中...")
                try:
                    axial_results = self._perform_axial_analysis(network)
                    result["axial_analysis"] = axial_results
                except Exception as e:
                    print(f"   ⚠️ 軸線分析スキップ: {e}")
                    result["axial_analysis"] = {"error": str(e)}
            
            if "integration" in include_analysis:
                print(f"   📈 統合評価実行中...")
                try:
                    integration_results = self._calculate_integration_metrics(result)
                    result["integration_metrics"] = integration_results
                except Exception as e:
                    print(f"   ⚠️ 統合評価スキップ: {e}")
                    result["integration_metrics"] = {"error": str(e)}
            
            # GraphML保存
            if settings.get("save_graphml", True):
                graphml_path = self.output_dir / f"{station_id}_network.graphml"
                try:
                    ox.save_graphml(network, str(graphml_path))
                    result["saved_graphml"] = str(graphml_path)
                    print(f"   💾 GraphML保存: {graphml_path.name}")
                except Exception as e:
                    print(f"   ⚠️ GraphML保存失敗: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"駅分析エラー ({station_id}): {e}")
            return {
                "error": True,
                "error_message": str(e)
            }
    
    def _calculate_network_metrics(self, network: nx.MultiDiGraph) -> Dict[str, Any]:
        """ネットワークから基本指標を計算"""
        try:
            import numpy as np
            
            node_count = network.number_of_nodes()
            edge_count = network.number_of_edges()
            
            if node_count == 0:
                return {"error": "空のネットワーク"}
            
            # 基本指標の計算
            degrees = dict(network.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            max_degree = max(degrees.values()) if degrees else 0
            
            # 密度計算
            density = nx.density(network)
            
            # 接続性の確認
            is_connected = nx.is_connected(network.to_undirected()) if node_count > 0 else False
            
            # 最大連結成分
            if not is_connected and node_count > 0:
                largest_cc = max(nx.connected_components(network.to_undirected()), key=len)
                largest_component_size = len(largest_cc)
            else:
                largest_component_size = node_count
            
            # 道路総延長の計算（概算）
            total_length = 0
            for u, v, data in network.edges(data=True):
                length = data.get('length', 0)
                if length > 0:
                    total_length += length
            
            # エリア計算（バウンディングボックスから概算）
            coords = []
            for node, data in network.nodes(data=True):
                if 'x' in data and 'y' in data:
                    coords.append([data['x'], data['y']])
            
            area_km2 = 0
            if coords:
                coords = np.array(coords)
                # 簡易面積計算（矩形近似）
                x_range = coords[:, 0].max() - coords[:, 0].min()
                y_range = coords[:, 1].max() - coords[:, 1].min()
                # 緯度経度を概算でメートルに変換（東京付近）
                x_meters = x_range * 111000 * np.cos(np.radians(35.6))
                y_meters = y_range * 111000
                area_km2 = (x_meters * y_meters) / 1000000  # km^2
            
            # Space Syntax風の指標計算
            # α指数（循環性）: 実際の回路数 / 最大可能回路数
            alpha_index = max(0, edge_count - node_count + 1) if node_count > 2 else 0
            
            # β指数（複雑性）: エッジ数 / ノード数
            beta_index = edge_count / node_count if node_count > 0 else 0
            
            # γ指数（接続性）: 実際のエッジ数 / 最大可能エッジ数
            max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
            gamma_index = edge_count / max_edges if max_edges > 0 else 0
            
            # 道路密度
            road_density = total_length / 1000 / area_km2 if area_km2 > 0 else 0  # km/km^2
            
            return {
                "analysis_status": "success",
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree": avg_degree,
                "max_degree": max_degree,
                "density": density,
                "is_connected": is_connected,
                "largest_component_size": largest_component_size,
                "total_length_m": total_length,
                "area_km2": area_km2,
                "alpha_index": alpha_index,
                "beta_index": beta_index,
                "gamma_index": gamma_index,
                "road_density": road_density
            }
            
        except Exception as e:
            logger.error(f"ネットワーク指標計算エラー: {e}")
            return {"error": str(e)}
    
    def _get_or_load_network(self, station: Dict[str, Any], settings: Dict[str, Any]) -> Optional[nx.MultiDiGraph]:
        """ネットワークを取得または既存ファイルから読み込み"""
        graphml_path = station.get("graphml_path")
        
        # 既存のGraphMLファイルがある場合は読み込み
        if graphml_path and Path(graphml_path).exists():
            print(f"   📂 GraphMLファイル読み込み: {graphml_path}")
            try:
                return ox.load_graphml(graphml_path)
            except Exception as e:
                print(f"   ⚠️ GraphML読み込み失敗: {e}, ネットワーク取得に切り替え")
        
        # ネットワークを新規取得
        radius = settings.get("radius_meters", 800)
        network_type = settings.get("network_type", "drive")
        
        coordinates = station.get("coordinates")
        location = station.get("location")
        
        try:
            if coordinates:
                # 座標指定での取得
                print(f"   🌐 座標からネットワーク取得 (半径{radius}m)...")
                lat, lon = coordinates
                network = ox.graph_from_point((lat, lon), dist=radius, network_type=network_type)
            elif location:
                # 地名指定での取得
                print(f"   🔍 地名からネットワーク取得: {location} (半径{radius}m)...")
                network = ox.graph_from_address(location, dist=radius, network_type=network_type)
            else:
                print(f"   ❌ 座標も地名も指定されていません")
                return None
            
            # ネットワークの前処理
            network = ox.add_edge_speeds(network)
            network = ox.add_edge_travel_times(network)
            
            print(f"   ✅ ネットワーク取得完了: {network.number_of_nodes()}ノード, {network.number_of_edges()}エッジ")
            return network
            
        except Exception as e:
            print(f"   ❌ ネットワーク取得エラー: {e}")
            return None
    
    def _perform_axial_analysis(self, network: nx.MultiDiGraph) -> Dict[str, Any]:
        """軸線分析の実行"""
        try:
            from space_syntax_analyzer.core.axial import AxialAnalyzer
            
            axial_analyzer = AxialAnalyzer()
            axial_results = axial_analyzer.calculate_axial_summary(network)
            
            return axial_results
            
        except Exception as e:
            logger.warning(f"軸線分析エラー: {e}")
            return {"error": str(e)}
    
    def _calculate_integration_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """統合評価指標の計算"""
        try:
            basic_analysis = result.get("basic_analysis", {})
            network_data = result.get("network_data", {})
            station_info = result.get("station_info", {})
            
            # 基本指標
            node_count = network_data.get("node_count", 0)
            edge_count = network_data.get("edge_count", 0)
            
            # 密度指標
            if node_count > 1:
                density = edge_count / (node_count * (node_count - 1) / 2) if node_count > 1 else 0
            else:
                density = 0
            
            # アクセシビリティスコア（駅周辺特化）
            accessibility_score = min(node_count / 100 * 50 + density * 50, 100)
            
            # 接続性スコア
            avg_degree = (2 * edge_count / node_count) if node_count > 0 else 0
            connectivity_score = min(avg_degree * 10, 100)
            
            # 駅周辺総合スコア
            station_score = (accessibility_score + connectivity_score) / 2
            
            # 評価レベル
            if station_score >= 80:
                evaluation_level = "A - 非常に良好"
            elif station_score >= 65:
                evaluation_level = "B - 良好"
            elif station_score >= 50:
                evaluation_level = "C - 標準"
            elif station_score >= 35:
                evaluation_level = "D - 改善の余地あり"
            else:
                evaluation_level = "E - 大幅改善必要"
            
            return {
                "accessibility_score": accessibility_score,
                "connectivity_score": connectivity_score,
                "station_score": station_score,
                "evaluation_level": evaluation_level,
                "network_density": density,
                "average_degree": avg_degree,
                "analysis_radius": result.get("settings", {}).get("radius_meters", 800)
            }
            
        except Exception as e:
            logger.warning(f"統合評価計算エラー: {e}")
            return {"error": str(e)}
    
    def _save_station_visualizations(self, station: Dict[str, Any], result: Dict[str, Any]):
        """駅の可視化を保存"""
        try:
            station_id = station.get("id")
            station_name = station.get("name")
            
            # ネットワークの再取得（可視化用）
            settings = result.get("settings", {})
            network = self._get_or_load_network(station, settings)
            
            if network is None:
                print(f"   ⚠️ 可視化用ネットワーク取得失敗")
                return
            
            print(f"   📊 可視化保存中...")
            
            # 背景地図の利用可否
            background_map = (settings.get("background_map", True) and 
                            self.visualizer.contextily_available and 
                            self.visualizer.geopandas_available)
            
            # 1. 基本ネットワーク図
            network_path = self.output_dir / f"{station_id}_network.png"
            success = self.visualizer.save_network_graph(
                network,
                str(network_path),
                title=f"{station_name} 周辺道路ネットワーク",
                show_basemap=background_map,
                basemap_alpha=0.6,
                edge_color="darkblue",
                node_color="red"
            )
            if success:
                print(f"   ✅ ネットワーク図: {network_path.name}")
            
            # 2. 軸線分析図（利用可能な場合）
            axial_analysis = result.get("axial_analysis", {})
            if axial_analysis and not axial_analysis.get("error"):
                axial_path = self.output_dir / f"{station_id}_axial.png"
                success = self.visualizer.save_axial_lines_only(
                    axial_analysis,
                    str(axial_path),
                    title=f"{station_name} 軸線分析",
                    show_basemap=background_map,
                    basemap_alpha=0.6
                )
                if success:
                    print(f"   ✅ 軸線分析図: {axial_path.name}")
            
        except Exception as e:
            print(f"   ⚠️ 可視化保存エラー: {e}")
            logger.warning(f"可視化保存エラー ({station.get('id')}): {e}")
    
    def _generate_comparative_analysis(self, successful_analyses: List[Tuple[str, str, Dict[str, Any]]]):
        """駅間比較分析の生成"""
        try:
            print("   📊 比較チャート生成中...")
            
            # 比較データの準備
            comparison_data = []
            for station_id, station_name, result in successful_analyses:
                network_data = result.get("network_data", {})
                integration_metrics = result.get("integration_metrics", {})
                
                comparison_data.append({
                    "駅ID": station_id,
                    "駅名": station_name,
                    "ノード数": network_data.get("node_count", 0),
                    "エッジ数": network_data.get("edge_count", 0),
                    "アクセシビリティスコア": integration_metrics.get("accessibility_score", 0),
                    "接続性スコア": integration_metrics.get("connectivity_score", 0),
                    "総合スコア": integration_metrics.get("station_score", 0),
                    "評価レベル": integration_metrics.get("evaluation_level", "不明")
                })
            
            # DataFrameとして保存
            df = pd.DataFrame(comparison_data)
            csv_path = self.output_dir / "station_comparison.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   ✅ 比較データ: {csv_path.name}")
            
            # 比較チャートの生成
            self._create_comparison_charts(df)
            
        except Exception as e:
            print(f"   ⚠️ 比較分析エラー: {e}")
            logger.warning(f"比較分析エラー: {e}")
    
    def _create_comparison_charts(self, df: pd.DataFrame):
        """比較チャートの作成"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if df.empty:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            if JAPANESE_FONT_AVAILABLE:
                fig.suptitle("駅周辺ネットワーク比較分析", fontsize=16, fontweight="bold")
            else:
                fig.suptitle("Station Network Comparison Analysis", fontsize=16, fontweight="bold")
            
            # 1. ネットワーク規模比較
            x_pos = np.arange(len(df))
            width = 0.35
            
            ax1.bar(x_pos - width/2, df["ノード数"], width, label="ノード数", color="#3498DB")
            ax1.bar(x_pos + width/2, df["エッジ数"], width, label="エッジ数", color="#E74C3C")
            ax1.set_title("ネットワーク規模比較" if JAPANESE_FONT_AVAILABLE else "Network Size Comparison")
            ax1.set_xlabel("駅" if JAPANESE_FONT_AVAILABLE else "Station")
            ax1.set_ylabel("数量" if JAPANESE_FONT_AVAILABLE else "Count")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(df["駅名"], rotation=45)
            ax1.legend()
            
            # 2. スコア比較
            ax2.bar(df["駅名"], df["総合スコア"], color="#2ECC71")
            ax2.set_title("総合スコア比較" if JAPANESE_FONT_AVAILABLE else "Overall Score Comparison")
            ax2.set_xlabel("駅" if JAPANESE_FONT_AVAILABLE else "Station")
            ax2.set_ylabel("スコア" if JAPANESE_FONT_AVAILABLE else "Score")
            ax2.set_ylim(0, 100)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 3. アクセシビリティ vs 接続性
            scatter = ax3.scatter(df["アクセシビリティスコア"], df["接続性スコア"], 
                                c=df["総合スコア"], cmap="viridis", s=100, alpha=0.7)
            ax3.set_title("アクセシビリティ vs 接続性" if JAPANESE_FONT_AVAILABLE else "Accessibility vs Connectivity")
            ax3.set_xlabel("アクセシビリティスコア" if JAPANESE_FONT_AVAILABLE else "Accessibility Score")
            ax3.set_ylabel("接続性スコア" if JAPANESE_FONT_AVAILABLE else "Connectivity Score")
            
            # 駅名をポイントに追加
            for i, row in df.iterrows():
                ax3.annotate(row["駅名"], (row["アクセシビリティスコア"], row["接続性スコア"]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.colorbar(scatter, ax=ax3, label="総合スコア" if JAPANESE_FONT_AVAILABLE else "Overall Score")
            
            # 4. 評価レベル分布
            level_counts = df["評価レベル"].value_counts()
            ax4.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title("評価レベル分布" if JAPANESE_FONT_AVAILABLE else "Evaluation Level Distribution")
            
            plt.tight_layout()
            
            chart_path = self.output_dir / "station_comparison_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()
            
            print(f"   ✅ 比較チャート: {chart_path.name}")
            
        except Exception as e:
            print(f"   ⚠️ 比較チャート作成エラー: {e}")
            logger.warning(f"比較チャート作成エラー: {e}")
    
    def _generate_station_report(self, successful_analyses: List[Tuple[str, str, Dict[str, Any]]]):
        """駅分析統合レポートの生成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"station_analysis_report_{timestamp}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # レポートヘッダー
                f.write("# 駅周辺道路ネットワーク分析レポート\n\n")
                f.write(f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
                
                # 分析概要
                f.write("## 📋 分析概要\n\n")
                f.write(f"- **分析対象駅数**: {len(successful_analyses)}駅\n")
                f.write(f"- **分析半径**: {self.config.get('analysis_settings', {}).get('radius_meters', 800)}m\n")
                f.write(f"- **ネットワーク種別**: {self.config.get('analysis_settings', {}).get('network_type', 'drive')}\n\n")
                
                # 駅別詳細結果
                f.write("## 🚉 駅別分析結果\n\n")
                
                for station_id, station_name, result in successful_analyses:
                    station_info = result.get("station_info", {})
                    network_data = result.get("network_data", {})
                    integration_metrics = result.get("integration_metrics", {})
                    
                    f.write(f"### {station_name}\n\n")
                    f.write(f"**駅ID**: {station_id}\n\n")
                    f.write(f"**説明**: {station_info.get('description', '説明なし')}\n\n")
                    
                    # ネットワーク基本情報
                    f.write("#### ネットワーク基本情報\n\n")
                    f.write(f"- ノード数: {network_data.get('node_count', 0):,}\n")
                    f.write(f"- エッジ数: {network_data.get('edge_count', 0):,}\n")
                    f.write(f"- ネットワーク密度: {integration_metrics.get('network_density', 0):.4f}\n")
                    f.write(f"- 平均次数: {integration_metrics.get('average_degree', 0):.2f}\n\n")
                    
                    # 評価指標
                    if integration_metrics and not integration_metrics.get("error"):
                        f.write("#### 評価指標\n\n")
                        f.write(f"- アクセシビリティスコア: {integration_metrics.get('accessibility_score', 0):.1f}/100\n")
                        f.write(f"- 接続性スコア: {integration_metrics.get('connectivity_score', 0):.1f}/100\n")
                        f.write(f"- **総合スコア**: {integration_metrics.get('station_score', 0):.1f}/100\n")
                        f.write(f"- **評価レベル**: {integration_metrics.get('evaluation_level', '不明')}\n\n")
                    
                    # 軸線分析結果
                    axial_analysis = result.get("axial_analysis", {})
                    if axial_analysis and not axial_analysis.get("error"):
                        network_metrics = axial_analysis.get("network_metrics", {})
                        if network_metrics:
                            f.write("#### 軸線分析結果\n\n")
                            f.write(f"- 軸線数: {network_metrics.get('axial_lines', 0)}\n")
                            f.write(f"- 軸線接続数: {network_metrics.get('axial_connections', 0)}\n")
                            f.write(f"- 格子度: {network_metrics.get('grid_axiality', 0):.3f}\n\n")
                
                # 比較分析
                if len(successful_analyses) > 1:
                    f.write("## 📊 駅間比較分析\n\n")
                    
                    # ランキング
                    sorted_stations = sorted(successful_analyses, 
                                           key=lambda x: x[2].get("integration_metrics", {}).get("station_score", 0), 
                                           reverse=True)
                    
                    f.write("### 総合スコアランキング\n\n")
                    for i, (station_id, station_name, result) in enumerate(sorted_stations, 1):
                        score = result.get("integration_metrics", {}).get("station_score", 0)
                        level = result.get("integration_metrics", {}).get("evaluation_level", "不明")
                        f.write(f"{i}. **{station_name}**: {score:.1f}点 ({level})\n")
                    
                    f.write("\n")
                    
                    # 特徴分析
                    f.write("### 特徴分析\n\n")
                    
                    # 最高スコア駅
                    best_station = sorted_stations[0]
                    f.write(f"**最も優秀な駅**: {best_station[1]}\n")
                    f.write(f"- 特徴: 高いアクセシビリティと接続性を兼ね備えた駅周辺環境\n\n")
                    
                    # ネットワーク規模最大
                    largest_network = max(successful_analyses, 
                                        key=lambda x: x[2].get("network_data", {}).get("node_count", 0))
                    f.write(f"**最大ネットワーク**: {largest_network[1]}\n")
                    f.write(f"- ノード数: {largest_network[2].get('network_data', {}).get('node_count', 0):,}\n\n")
                
                # 推奨事項
                f.write("## 💡 推奨事項\n\n")
                f.write("### 都市計画への応用\n")
                f.write("- 総合スコアの高い駅は、歩行者・自転車利用の促進に適している\n")
                f.write("- 接続性の低い駅周辺では、新たな道路整備を検討\n")
                f.write("- アクセシビリティの低い駅では、既存道路の改良を検討\n\n")
                
                f.write("### 交通政策への示唆\n")
                f.write("- 高スコア駅: 公共交通との結節機能強化\n")
                f.write("- 低スコア駅: 駅前再開発や道路網整備の優先実施\n")
                f.write("- 軸線分析結果を活用した効率的な交通流設計\n\n")
                
                # メタデータ
                f.write("---\n\n")
                f.write("## 📋 分析メタデータ\n\n")
                f.write("- **分析手法**: Space Syntax Analysis + Network Analysis\n")
                f.write("- **データソース**: OpenStreetMap\n")
                f.write("- **分析ツール**: OSMnx, NetworkX, Space Syntax Analyzer\n")
                f.write(f"- **設定ファイル**: {self.config_path}\n")
                f.write(f"- **出力ディレクトリ**: {self.output_dir}\n")
            
            print(f"   ✅ 統合レポート: {report_path.name}")
            
        except Exception as e:
            print(f"   ⚠️ レポート生成エラー: {e}")
            logger.warning(f"レポート生成エラー: {e}")


def main():
    """メイン関数"""
    print("🚉 駅周辺道路ネットワーク分析デモ")
    print("="*80)
    
    # 設定ファイルのパス確認
    config_files = ["station_config.json", "config/station_config.json", "examples/station_config.json"]
    config_path = None
    
    for path in config_files:
        if Path(path).exists():
            config_path = path
            break
    
    if not config_path:
        # デフォルトパスで新規作成
        config_path = "station_config.json"
    
    try:
        # 分析器の初期化
        analyzer = StationNetworkAnalyzer(config_path)
        
        # 分析実行
        results = analyzer.analyze_all_stations()
        
        print(f"\n📁 結果は {analyzer.output_dir} に保存されました")
        print("💡 生成されたファイル:")
        print("   - 各駅のネットワーク図 (PNG)")
        print("   - 軸線分析図 (PNG)")
        print("   - GraphMLネットワークファイル")
        print("   - 駅間比較データ (CSV)")
        print("   - 比較チャート (PNG)")
        print("   - 統合分析レポート (Markdown)")
        
    except KeyboardInterrupt:
        print("\n👋 ユーザーによって中断されました")
    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
        print(f"❌ 予期しないエラー: {e}")


if __name__ == "__main__":
    main()