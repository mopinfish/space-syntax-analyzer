"""
軸線分析モジュール - AxialAnalyzer

スペースシンタックス理論のAxial Analysisを実装します。
軸線マップの作成からIntegration Value計算まで一貫して行います。
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge

logger = logging.getLogger(__name__)


class AxialAnalyzer:
    """
    軸線分析を行うクラス

    スペースシンタックス理論に基づく軸線分析の全機能を提供します。
    """

    def __init__(self, simplification_tolerance: float = 1.0) -> None:
        """
        AxialAnalyzerを初期化

        Args:
            simplification_tolerance: 軸線簡略化の許容誤差（メートル）
        """
        self.simplification_tolerance = simplification_tolerance

    def create_axial_map(self, network: nx.Graph) -> nx.Graph:
        """
        道路ネットワークから軸線マップを作成

        Args:
            network: 道路ネットワークグラフ

        Returns:
            軸線グラフ
        """
        try:
            logger.info("軸線マップの作成を開始")

            # 1. 凸空間の分割
            convex_spaces = self._create_convex_spaces(network)

            # 2. 軸線の生成
            axial_lines = self._generate_axial_lines(convex_spaces, network)

            # 3. 軸線グラフの構築
            axial_graph = self._build_axial_graph(axial_lines)

            # 4. 軸線の最適化
            optimized_graph = self._optimize_axial_map(axial_graph)

            logger.info(f"軸線マップ作成完了: {optimized_graph.number_of_nodes()}本の軸線")
            return optimized_graph

        except Exception as e:
            logger.error(f"軸線マップ作成エラー: {e}")
            raise

    def _create_convex_spaces(self, network: nx.Graph) -> list[Polygon]:
        """
        ネットワークから凸空間を作成

        Args:
            network: 道路ネットワーク

        Returns:
            凸空間のリスト
        """
        try:
            # エッジから線分を作成
            lines = []
            for u, v, _data in network.edges(data=True):
                u_data = network.nodes[u]
                v_data = network.nodes[v]

                if all(key in u_data for key in ["x", "y"]) and \
                   all(key in v_data for key in ["x", "y"]):
                    line = LineString([(u_data["x"], u_data["y"]),
                                     (v_data["x"], v_data["y"])])
                    lines.append(line)

            if not lines:
                return []

            # 線分の結合によるポリゴン生成
            linemerge(lines)

            # 凸空間の近似（簡易版）
            # 実際の実装では、より高度なアルゴリズムが必要
            convex_spaces = []

            # ネットワークの外接矩形を基本領域として使用
            bounds = self._get_network_bounds(network)
            if bounds:
                # 基本的な分割を実行（グリッド状分割の簡易版）
                min_x, min_y, max_x, max_y = bounds
                grid_size = min((max_x - min_x) / 10, (max_y - min_y) / 10)

                for i in range(10):
                    for j in range(10):
                        x1 = min_x + i * grid_size
                        y1 = min_y + j * grid_size
                        x2 = min_x + (i + 1) * grid_size
                        y2 = min_y + (j + 1) * grid_size

                        cell = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                        convex_spaces.append(cell)

            return convex_spaces

        except Exception as e:
            logger.warning(f"凸空間作成エラー: {e}")
            return []

    def _generate_axial_lines(self, convex_spaces: list[Polygon],
                            network: nx.Graph) -> list[LineString]:
        """
        凸空間を貫通する軸線を生成

        Args:
            convex_spaces: 凸空間のリスト
            network: 道路ネットワーク

        Returns:
            軸線のリスト
        """
        try:
            axial_lines = []

            if not convex_spaces:
                # 凸空間がない場合は、道路エッジから直接軸線を生成
                return self._generate_axial_lines_from_edges(network)

            # 各凸空間に対して最適な軸線を生成
            for space in convex_spaces:
                if space.is_valid and not space.is_empty:
                    # 凸空間の最長軸線を計算
                    longest_line = self._find_longest_line_in_polygon(space)
                    if longest_line:
                        axial_lines.append(longest_line)

            # 重複する軸線の統合
            merged_lines = self._merge_similar_lines(axial_lines)

            return merged_lines

        except Exception as e:
            logger.warning(f"軸線生成エラー: {e}")
            return self._generate_axial_lines_from_edges(network)

    def _generate_axial_lines_from_edges(self, network: nx.Graph) -> list[LineString]:
        """
        道路エッジから軸線を生成（フォールバック）

        Args:
            network: 道路ネットワーク

        Returns:
            軸線のリスト
        """
        axial_lines = []

        for u, v, _data in network.edges(data=True):
            u_data = network.nodes[u]
            v_data = network.nodes[v]

            if all(key in u_data for key in ["x", "y"]) and \
               all(key in v_data for key in ["x", "y"]):
                line = LineString([(u_data["x"], u_data["y"]),
                                 (v_data["x"], v_data["y"])])
                axial_lines.append(line)

        return axial_lines

    def _find_longest_line_in_polygon(self, polygon: Polygon) -> LineString | None:
        """
        ポリゴン内の最長線分を見つける

        Args:
            polygon: 対象ポリゴン

        Returns:
            最長線分
        """
        try:
            bounds = polygon.bounds
            min_x, min_y, max_x, max_y = bounds

            # 対角線を軸線の候補とする
            diagonal1 = LineString([(min_x, min_y), (max_x, max_y)])
            diagonal2 = LineString([(min_x, max_y), (max_x, min_y)])

            # より長い対角線を選択
            if diagonal1.length > diagonal2.length:
                return diagonal1
            else:
                return diagonal2

        except Exception:
            return None

    def _merge_similar_lines(self, lines: list[LineString]) -> list[LineString]:
        """
        類似する軸線を統合

        Args:
            lines: 軸線のリスト

        Returns:
            統合後の軸線リスト
        """
        if not lines:
            return []

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            candidates = [line1]
            used.add(i)

            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue

                # 角度と距離の類似性をチェック
                if self._lines_are_similar(line1, line2):
                    candidates.append(line2)
                    used.add(j)

            # 候補線分を統合
            if len(candidates) > 1:
                merged_line = self._merge_line_segments(candidates)
                if merged_line:
                    merged.append(merged_line)
            else:
                merged.append(line1)

        return merged

    def _lines_are_similar(self, line1: LineString, line2: LineString,
                          angle_threshold: float = 15.0,
                          distance_threshold: float = 50.0) -> bool:
        """
        2つの線分が類似しているかチェック

        Args:
            line1, line2: 比較する線分
            angle_threshold: 角度の閾値（度）
            distance_threshold: 距離の閾値（メートル）

        Returns:
            類似しているかどうか
        """
        try:
            # 角度の計算
            angle1 = self._calculate_line_angle(line1)
            angle2 = self._calculate_line_angle(line2)

            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 180 - angle_diff)  # 0-90度の範囲に正規化

            # 距離の計算
            distance = line1.distance(line2)

            return angle_diff < angle_threshold and distance < distance_threshold

        except Exception:
            return False

    def _calculate_line_angle(self, line: LineString) -> float:
        """
        線分の角度を計算（度）

        Args:
            line: 線分

        Returns:
            角度（0-180度）
        """
        coords = list(line.coords)
        if len(coords) < 2:
            return 0.0

        x1, y1 = coords[0]
        x2, y2 = coords[-1]

        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        # 0-180度の範囲に正規化
        return abs(angle_deg) % 180

    def _merge_line_segments(self, lines: list[LineString]) -> LineString | None:
        """
        複数の線分を統合

        Args:
            lines: 統合する線分のリスト

        Returns:
            統合された線分
        """
        try:
            # 線分の結合を試行
            merged = linemerge(lines)

            if isinstance(merged, LineString):
                return merged
            else:
                # 結合に失敗した場合は最長の線分を返す
                return max(lines, key=lambda x: x.length)

        except Exception:
            return lines[0] if lines else None

    def _build_axial_graph(self, axial_lines: list[LineString]) -> nx.Graph:
        """
        軸線からグラフを構築

        Args:
            axial_lines: 軸線のリスト

        Returns:
            軸線グラフ
        """
        graph = nx.Graph()

        # 軸線をノードとして追加
        for i, line in enumerate(axial_lines):
            graph.add_node(i, geometry=line, length=line.length)

        # 交差する軸線間にエッジを追加
        for i in range(len(axial_lines)):
            for j in range(i + 1, len(axial_lines)):
                if axial_lines[i].intersects(axial_lines[j]):
                    graph.add_edge(i, j)

        return graph

    def _optimize_axial_map(self, axial_graph: nx.Graph) -> nx.Graph:
        """
        軸線マップの最適化（余分な軸線の除去）

        Args:
            axial_graph: 原始軸線グラフ

        Returns:
            最適化された軸線グラフ
        """
        optimized = axial_graph.copy()

        # 度数1のノード（端点のみ）を除去する処理
        removed_nodes = []
        for node in list(optimized.nodes()):
            if optimized.degree[node] <= 1 and self._should_remove_axial_line(optimized, node):
                # 隣接する軸線との関係を考慮して除去判定
                removed_nodes.append(node)

        optimized.remove_nodes_from(removed_nodes)

        logger.info(f"軸線最適化: {len(removed_nodes)}本の軸線を除去")

        return optimized

    def _should_remove_axial_line(self, graph: nx.Graph, node: int) -> bool:
        """
        軸線を除去すべきかどうかを判定

        Args:
            graph: 軸線グラフ
            node: 判定対象ノード

        Returns:
            除去すべきかどうか
        """
        # 度数が1以下の場合は除去候補
        if graph.degree[node] <= 1:
            node_data = graph.nodes[node]
            line_length = node_data.get("length", 0)

            # 短すぎる軸線は除去
            if line_length < 50:  # 50m未満
                return True

        return False

    def calculate_integration_value(self, axial_graph: nx.Graph,
                                  radius: int | None = None) -> dict[int, float]:
        """
        Integration Valueを計算

        Args:
            axial_graph: 軸線グラフ
            radius: 解析半径（Noneの場合はGlobalレベル）

        Returns:
            各軸線のIntegration Value
        """
        try:
            integration_values = {}

            for node in axial_graph.nodes():
                # 深度計算
                depths = self._calculate_depths(axial_graph, node, radius)

                # Total Depthの計算
                total_depth = sum(depths.values())

                # Mean Depthの計算
                k = len(depths)
                mean_depth = 0 if k <= 1 else total_depth / (k - 1)

                # Relative Asymmetry (RA)の計算
                ra = self._calculate_relative_asymmetry(mean_depth, k)

                # Real Relative Asymmetry (RRA)の計算
                dk = self._calculate_dk(k)
                rra = ra / dk if dk > 0 else 0

                # Integration Valueの計算
                integration_value = 1 / rra if rra > 0 else 0

                integration_values[node] = integration_value

            return integration_values

        except Exception as e:
            logger.error(f"Integration Value計算エラー: {e}")
            return {}

    def _calculate_depths(self, graph: nx.Graph, start_node: int,
                         radius: int | None = None) -> dict[int, int]:
        """
        指定ノードから他の全ノードへの深度を計算

        Args:
            graph: 軸線グラフ
            start_node: 開始ノード
            radius: 解析半径

        Returns:
            各ノードへの深度
        """
        try:
            if radius is None:
                # Global分析: 全ノードへの最短経路
                path_lengths = nx.single_source_shortest_path_length(graph, start_node)
            else:
                # Local分析: 指定半径内のノードのみ
                path_lengths = nx.single_source_shortest_path_length(
                    graph, start_node, cutoff=radius
                )

            return path_lengths

        except Exception as e:
            logger.warning(f"深度計算エラー: {e}")
            return {start_node: 0}

    def _calculate_relative_asymmetry(self, mean_depth: float, k: int) -> float:
        """
        Relative Asymmetry (RA)を計算

        Args:
            mean_depth: 平均深度
            k: 軸線総数

        Returns:
            Relative Asymmetry
        """
        if k <= 2:
            return 0.0

        # RA = (MD - 1) / (k - 2)
        ra = (mean_depth - 1) / (k - 2)
        return max(0.0, ra)

    def _calculate_dk(self, k: int) -> float:
        """
        標準化因子Dkを計算

        Args:
            k: 軸線総数

        Returns:
            標準化因子Dk
        """
        if k <= 3:
            return 1.0

        # Dk = 2 * (k * log((k+2)/3) - 1) / ((k-1)(k-2))
        import math

        numerator = 2 * (k * math.log((k + 2) / 3) - 1)
        denominator = (k - 1) * (k - 2)

        if denominator == 0:
            return 1.0

        dk = numerator / denominator
        return max(0.001, dk)  # 0除算を避ける

    def calculate_axial_network_metrics(self, axial_graph: nx.Graph) -> dict[str, float]:
        """
        軸線ネットワークの形態指標を計算

        Args:
            axial_graph: 軸線グラフ

        Returns:
            形態指標
        """
        try:
            # 基本統計
            num_lines = axial_graph.number_of_nodes()  # 軸線数
            num_connections = axial_graph.number_of_edges()  # 接続数

            if num_lines == 0:
                return self._empty_axial_metrics()

            # アイランド数の計算（完全に囲まれた空間）
            num_islands = self._count_islands(axial_graph)

            # 格子度（Grid Axiality: GA）
            ga = self._calculate_grid_axiality(num_lines, num_islands)

            # 循環度（Axial Ringiness: AR）
            ar = self._calculate_axial_ringiness(num_lines, num_islands)

            # 分節度（Axial Articulation: AA）
            aa = self._calculate_axial_articulation(num_lines, num_islands)

            return {
                "axial_lines": num_lines,
                "axial_connections": num_connections,
                "axial_islands": num_islands,
                "grid_axiality": ga,
                "axial_ringiness": ar,
                "axial_articulation": aa,
            }

        except Exception as e:
            logger.error(f"軸線ネットワーク指標計算エラー: {e}")
            return self._empty_axial_metrics()

    def _count_islands(self, axial_graph: nx.Graph) -> int:
        """
        アイランド数（完全に囲まれた空間）を計算

        Args:
            axial_graph: 軸線グラフ

        Returns:
            アイランド数
        """
        try:
            # 簡易的なアイランド計算
            # 実際には、軸線が形成する閉領域を厳密に計算する必要がある

            # 基本的な回路数を近似として使用
            num_nodes = axial_graph.number_of_nodes()
            num_edges = axial_graph.number_of_edges()
            num_components = nx.number_connected_components(axial_graph)

            # オイラーの公式から面の数を推定
            # F = E - V + C + 1（外部面を含む）
            estimated_faces = num_edges - num_nodes + num_components + 1

            # 外部面を除外してアイランド数とする
            islands = max(0, estimated_faces - 1)

            return islands

        except Exception:
            return 0

    def _calculate_grid_axiality(self, num_lines: int, num_islands: int) -> float:
        """
        格子度（Grid Axiality）を計算

        Args:
            num_lines: 軸線数
            num_islands: アイランド数

        Returns:
            格子度（0-1）
        """
        if num_lines <= 0:
            return 0.0

        # GA = 4*I / L
        ga = (4 * num_islands) / num_lines
        return min(1.0, ga)  # 最大値を1に制限

    def _calculate_axial_ringiness(self, num_lines: int, num_islands: int) -> float:
        """
        循環度（Axial Ringiness）を計算

        Args:
            num_lines: 軸線数
            num_islands: アイランド数

        Returns:
            循環度
        """
        if num_lines <= 0:
            return 0.0

        # AR = I / L
        ar = num_islands / num_lines
        return ar

    def _calculate_axial_articulation(self, num_lines: int, num_islands: int) -> float:
        """
        分節度（Axial Articulation）を計算

        Args:
            num_lines: 軸線数
            num_islands: アイランド数

        Returns:
            分節度
        """
        if num_lines <= 0:
            return 0.0

        # AA = 街区数 / 軸線数
        # ここでは、アイランド数を街区数の近似として使用
        aa = num_islands / num_lines
        return aa

    def _get_network_bounds(self, network: nx.Graph) -> tuple[float, float, float, float] | None:
        """
        ネットワークの境界を取得

        Args:
            network: ネットワークグラフ

        Returns:
            境界座標 (min_x, min_y, max_x, max_y)
        """
        try:
            coordinates = []
            for _, data in network.nodes(data=True):
                if "x" in data and "y" in data:
                    coordinates.append((data["x"], data["y"]))

            if not coordinates:
                return None

            coords_array = np.array(coordinates)
            min_x, min_y = coords_array.min(axis=0)
            max_x, max_y = coords_array.max(axis=0)

            return (float(min_x), float(min_y), float(max_x), float(max_y))

        except Exception:
            return None

    def _empty_axial_metrics(self) -> dict[str, float]:
        """空の軸線指標を返す"""
        return {
            "axial_lines": 0,
            "axial_connections": 0,
            "axial_islands": 0,
            "grid_axiality": 0.0,
            "axial_ringiness": 0.0,
            "axial_articulation": 0.0,
        }

    def analyze_local_integration(self, axial_graph: nx.Graph,
                                 radius: int = 3) -> dict[int, float]:
        """
        Local Integration分析（指定半径内の分析）

        Args:
            axial_graph: 軸線グラフ
            radius: 分析半径

        Returns:
            Local Integration Value
        """
        return self.calculate_integration_value(axial_graph, radius)

    def analyze_global_integration(self, axial_graph: nx.Graph) -> dict[int, float]:
        """
        Global Integration分析（全範囲分析）

        Args:
            axial_graph: 軸線グラフ

        Returns:
            Global Integration Value
        """
        return self.calculate_integration_value(axial_graph, None)

    def calculate_axial_summary(self, network: nx.Graph) -> dict[str, Any]:
        """
        軸線分析の包括的実行

        Args:
            network: 道路ネットワーク

        Returns:
            軸線分析の全結果
        """
        try:
            # 軸線マップ作成
            axial_map = self.create_axial_map(network)

            # Integration Value計算
            global_integration = self.analyze_global_integration(axial_map)
            local_integration = self.analyze_local_integration(axial_map, radius=3)

            # 形態指標計算
            network_metrics = self.calculate_axial_network_metrics(axial_map)

            # 統計計算
            integration_stats = self._calculate_integration_statistics(global_integration)

            return {
                "axial_map": axial_map,
                "global_integration": global_integration,
                "local_integration": local_integration,
                "network_metrics": network_metrics,
                "integration_statistics": integration_stats,
            }

        except Exception as e:
            logger.error(f"軸線分析エラー: {e}")
            return {
                "axial_map": nx.Graph(),
                "global_integration": {},
                "local_integration": {},
                "network_metrics": self._empty_axial_metrics(),
                "integration_statistics": {},
            }

    def _calculate_integration_statistics(self, integration_values: dict[int, float]) -> dict[str, float]:
        """
        Integration Valueの統計を計算

        Args:
            integration_values: Integration Valueの辞書

        Returns:
            統計指標
        """
        if not integration_values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        values = list(integration_values.values())

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
