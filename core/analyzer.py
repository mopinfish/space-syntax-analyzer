"""
メインの分析クラス - SpaceSyntaxAnalyzer

このモジュールは、スペースシンタックス分析の統合インターフェースを提供します。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon

from .metrics import SpaceSyntaxMetrics
from .network import NetworkManager
from .visualization import NetworkVisualizer


class SpaceSyntaxAnalyzer:
    """
    スペースシンタックス分析を行うメインクラス
    
    このクラスは、道路ネットワークの取得から分析、可視化まで
    一連の処理を統合的に行います。
    """
    
    def __init__(
        self,
        width_threshold: float = 4.0,
        network_type: str = "drive",
        crs: str = "EPSG:4326"
    ) -> None:
        """
        SpaceSyntaxAnalyzerを初期化
        
        Args:
            width_threshold: 道路幅員の閾値（メートル）。この値以上を主要道路とする
            network_type: OSMnxのネットワークタイプ（'drive', 'walk', 'bike'）
            crs: 座標参照系
        """
        self.width_threshold = width_threshold
        self.network_type = network_type
        self.crs = crs
        
        self.network_manager = NetworkManager(
            width_threshold=width_threshold,
            network_type=network_type,
            crs=crs
        )
        self.metrics_calculator = SpaceSyntaxMetrics()
        self.visualizer = NetworkVisualizer()
        
    def get_network(
        self,
        location: Union[str, Tuple[float, float, float, float], Polygon],
        network_filter: str = "all"
    ) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        """
        指定された場所の道路ネットワークを取得
        
        Args:
            location: 場所の指定（地名、bbox座標、ポリゴン）
            network_filter: ネットワークフィルター（'major', 'all', 'both'）
            
        Returns:
            Tuple[主要道路ネットワーク, 全道路ネットワーク]
        """
        return self.network_manager.get_network(location, network_filter)
    
    def analyze(
        self,
        major_network: nx.MultiDiGraph,
        full_network: Optional[nx.MultiDiGraph] = None,
        area_ha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        スペースシンタックス分析を実行
        
        Args:
            major_network: 主要道路ネットワーク
            full_network: 全道路ネットワーク（オプション）
            area_ha: 分析対象エリアの面積（ヘクタール）
            
        Returns:
            分析結果辞書
        """
        results = {
            "major_network": self.metrics_calculator.calculate_all_metrics(
                major_network, area_ha
            )
        }
        
        if full_network is not None:
            results["full_network"] = self.metrics_calculator.calculate_all_metrics(
                full_network, area_ha
            )
            
        return results
    
    def analyze_place(
        self,
        location: Union[str, Tuple[float, float, float, float], Polygon],
        return_networks: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Tuple[nx.MultiDiGraph, nx.MultiDiGraph]]]:
        """
        場所を指定してワンステップで分析を実行
        
        Args:
            location: 分析対象の場所
            return_networks: ネットワークも返すかどうか
            
        Returns:
            分析結果、または分析結果とネットワークのタプル
        """
        # ネットワーク取得
        major_network, full_network = self.get_network(location, "both")
        
        # 面積計算
        area_ha = self.network_manager.calculate_area_ha(major_network)
        
        # 分析実行
        results = self.analyze(major_network, full_network, area_ha)
        
        if return_networks:
            return results, (major_network, full_network)
        return results
    
    def visualize(
        self,
        major_network: nx.MultiDiGraph,
        full_network: Optional[nx.MultiDiGraph] = None,
        results: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        ネットワークと分析結果を可視化
        
        Args:
            major_network: 主要道路ネットワーク
            full_network: 全道路ネットワーク（オプション）
            results: 分析結果（オプション）
            save_path: 保存パス（オプション）
        """
        self.visualizer.plot_network_comparison(
            major_network, full_network, results, save_path
        )
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format_type: str = "csv"
    ) -> None:
        """
        分析結果をファイルに出力
        
        Args:
            results: 分析結果
            output_path: 出力パス
            format_type: 出力形式（'csv', 'excel', 'json'）
        """
        if format_type.lower() == "csv":
            self._export_to_csv(results, output_path)
        elif format_type.lower() == "excel":
            self._export_to_excel(results, output_path)
        elif format_type.lower() == "json":
            self._export_to_json(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_to_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """CSV形式でエクスポート"""
        df_list = []
        
        for network_type, metrics in results.items():
            row = {"network_type": network_type}
            row.update(metrics)
            df_list.append(row)
            
        df = pd.DataFrame(df_list)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
    def _export_to_excel(self, results: Dict[str, Any], output_path: str) -> None:
        """Excel形式でエクスポート"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for network_type, metrics in results.items():
                df = pd.DataFrame([metrics])
                df.to_excel(writer, sheet_name=network_type, index=False)
                
    def _export_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        """JSON形式でエクスポート"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        location_name: str = "分析対象地域"
    ) -> str:
        """
        分析結果のレポートを生成
        
        Args:
            results: 分析結果
            location_name: 地域名
            
        Returns:
            分析レポート（文字列）
        """
        report = f"# {location_name} スペースシンタックス分析レポート\n\n"
        
        for network_type, metrics in results.items():
            network_name = "主要道路ネットワーク" if network_type == "major_network" else "全道路ネットワーク"
            report += f"## {network_name}\n\n"
            
            # 基本統計
            report += "### 基本統計\n"
            report += f"- ノード数: {metrics.get('nodes', 'N/A')}\n"
            report += f"- エッジ数: {metrics.get('edges', 'N/A')}\n"
            report += f"- 道路総延長: {metrics.get('total_length_m', 'N/A'):.1f}m\n\n"
            
            # 回遊性指標
            report += "### 回遊性指標\n"
            report += f"- 回路指数（μ）: {metrics.get('mu_index', 'N/A')}\n"
            report += f"- α指数: {metrics.get('alpha_index', 'N/A'):.1f}%\n"
            report += f"- β指数: {metrics.get('beta_index', 'N/A'):.2f}\n"
            report += f"- γ指数: {metrics.get('gamma_index', 'N/A'):.1f}%\n\n"
            
            # アクセス性指標
            report += "### アクセス性指標\n"
            report += f"- 平均最短距離（Di）: {metrics.get('avg_shortest_path', 'N/A'):.1f}m\n"
            report += f"- 道路密度（Dl）: {metrics.get('road_density', 'N/A'):.1f}m/ha\n"
            report += f"- 交差点密度（Dc）: {metrics.get('intersection_density', 'N/A'):.1f}n/ha\n\n"
            
            # 迂回性指標
            report += "### 迂回性指標\n"
            report += f"- 平均迂回率（A）: {metrics.get('avg_circuity', 'N/A'):.2f}\n\n"
            
        return report