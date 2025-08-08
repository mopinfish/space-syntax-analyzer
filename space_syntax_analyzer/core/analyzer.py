"""
メインの分析クラス - SpaceSyntaxAnalyzer

このモジュールは、スペースシンタックス分析の統合インターフェースを提供します。
"""

from __future__ import annotations

from typing import Any

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
        crs: str = "EPSG:4326",
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
            width_threshold=width_threshold, network_type=network_type, crs=crs
        )
        self.metrics_calculator = SpaceSyntaxMetrics()
        self.visualizer = NetworkVisualizer()

    def get_network(
        self,
        location: str | tuple[float, float, float, float] | Polygon,
        network_filter: str = "all",
    ) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
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
        full_network: nx.MultiDiGraph | None = None,
        area_ha: float | None = None,
    ) -> dict[str, Any]:
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
        location: str | tuple[float, float, float, float] | Polygon,
        return_networks: bool = False,
    ) -> (
        dict[str, Any] | tuple[dict[str, Any], tuple[nx.MultiDiGraph, nx.MultiDiGraph]]
    ):
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
        full_network: nx.MultiDiGraph | None = None,
        results: dict[str, Any] | None = None,
        save_path: str | None = None,
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
        self, results: dict[str, Any], output_path: str, format_type: str = "csv"
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

    def _export_to_csv(self, results: dict[str, Any], output_path: str) -> None:
        """CSV形式でエクスポート"""
        df_list = []

        for network_type, metrics in results.items():
            row = {"network_type": network_type}
            row.update(metrics)
            df_list.append(row)

        df = pd.DataFrame(df_list)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def _export_to_excel(self, results: dict[str, Any], output_path: str) -> None:
        """Excel形式でエクスポート"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for network_type, metrics in results.items():
                df = pd.DataFrame([metrics])
                df.to_excel(writer, sheet_name=network_type, index=False)

    def _export_to_json(self, results: dict[str, Any], output_path: str) -> None:
        """JSON形式でエクスポート"""
        import json

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def generate_report(
        self, results: dict[str, Any], location_name: str = "分析対象地域"
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
            network_name = (
                "主要道路ネットワーク"
                if network_type == "major_network"
                else "全道路ネットワーク"
            )
            report += f"## {network_name}\n\n"

            # 基本統計
            report += "### 基本統計\n"
            report += f"- ノード数: {metrics.get('nodes', 'N/A')}\n"
            report += f"- エッジ数: {metrics.get('edges', 'N/A')}\n"

            # total_length_m の安全な処理（修正箇所）
            total_length = metrics.get('total_length_m', 'N/A')
            if isinstance(total_length, int | float) and total_length != 'N/A':
                report += f"- 道路総延長: {total_length:.1f}m\n\n"
            else:
                report += f"- 道路総延長: {total_length}\n\n"

            # 回遊性指標
            report += "### 回遊性指標\n"
            report += f"- 回路指数（μ）: {metrics.get('mu_index', 'N/A')}\n"
            alpha_index = metrics.get('alpha_index', 'N/A')
            if isinstance(alpha_index, int | float):
                report += f"- α指数: {alpha_index:.1f}%\n"
            else:
                report += f"- α指数: {alpha_index}\n"

            beta_index = metrics.get('beta_index', 'N/A')
            if isinstance(beta_index, int | float):
                report += f"- β指数: {beta_index:.2f}\n"
            else:
                report += f"- β指数: {beta_index}\n"

            gamma_index = metrics.get('gamma_index', 'N/A')
            if isinstance(gamma_index, int | float):
                report += f"- γ指数: {gamma_index:.1f}%\n\n"
            else:
                report += f"- γ指数: {gamma_index}\n\n"

            # アクセス性指標
            report += "### アクセス性指標\n"
            avg_shortest_path = metrics.get('avg_shortest_path', 'N/A')
            if isinstance(avg_shortest_path, int | float):
                report += f"- 平均最短距離（Di）: {avg_shortest_path:.1f}m\n"
            else:
                report += f"- 平均最短距離（Di）: {avg_shortest_path}\n"

            road_density = metrics.get('road_density', 'N/A')
            if isinstance(road_density, int | float):
                report += f"- 道路密度（Dl）: {road_density:.1f}m/ha\n"
            else:
                report += f"- 道路密度（Dl）: {road_density}\n"

            intersection_density = metrics.get('intersection_density', 'N/A')
            if isinstance(intersection_density, int | float):
                report += f"- 交差点密度（Dc）: {intersection_density:.1f}n/ha\n\n"
            else:
                report += f"- 交差点密度（Dc）: {intersection_density}\n\n"

            # 迂回性指標
            report += "### 迂回性指標\n"
            avg_circuity = metrics.get('avg_circuity', 'N/A')
            if isinstance(avg_circuity, int | float):
                report += f"- 平均迂回率（A）: {avg_circuity:.2f}\n\n"
            else:
                report += f"- 平均迂回率（A）: {avg_circuity}\n\n"

        return report
