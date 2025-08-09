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
       道路ネットワークから軸線マップを作成（改善版）

       Args:
           network: 道路ネットワークグラフ

       Returns:
           軸線グラフ
       """
       try:
           logger.info("軸線マップの作成を開始")

           if not network or network.number_of_nodes() == 0:
               logger.warning("空のネットワークが提供されました")
               return nx.Graph()

           # 簡略化されたアプローチ：エッジベース軸線生成
           axial_graph = self._create_simplified_axial_map(network)

           if axial_graph.number_of_nodes() == 0:
               logger.warning("軸線生成に失敗、フォールバック方式を使用")
               axial_graph = self._create_fallback_axial_map(network)

           logger.info(f"軸線マップ作成完了: {axial_graph.number_of_nodes()}軸線")
           return axial_graph

       except Exception as e:
           logger.error(f"軸線マップ作成エラー: {e}")
           return nx.Graph()

   def _create_simplified_axial_map(self, network: nx.Graph) -> nx.Graph:
       """
       簡略化された軸線マップ作成

       Args:
           network: 道路ネットワーク

       Returns:
           軸線グラフ
       """
       axial_graph = nx.Graph()
       axial_id = 0

       # 直線的なエッジのグループ化
       edge_groups = self._group_collinear_edges(network)

       for group in edge_groups:
           if len(group) > 0:
               # 軸線としてノード追加
               axial_graph.add_node(axial_id, edges=group, length=self._calculate_group_length(group, network))
               axial_id += 1

       # 軸線間の接続を計算
       self._connect_axial_lines(axial_graph, network)

       return axial_graph

   def _group_collinear_edges(self, network: nx.Graph) -> list[list[tuple]]:
       """
       共線上のエッジをグループ化

       Args:
           network: 道路ネットワーク

       Returns:
           エッジグループのリスト
       """
       try:
           edges = list(network.edges())
           if not edges:
               return []

           visited = set()
           groups = []

           for edge in edges:
               if edge in visited:
                   continue

               # 新しいグループを開始
               group = [edge]
               visited.add(edge)

               # 同じ方向の隣接エッジを探す
               current_edge = edge
               while True:
                   next_edge = self._find_collinear_neighbor(current_edge, network, visited)
                   if next_edge:
                       group.append(next_edge)
                       visited.add(next_edge)
                       current_edge = next_edge
                   else:
                       break

               if len(group) > 0:
                   groups.append(group)

           return groups

       except Exception as e:
           logger.warning(f"エッジグループ化エラー: {e}")
           return []

   def _find_collinear_neighbor(self, edge: tuple, network: nx.Graph, visited: set) -> tuple | None:
       """
       共線上の隣接エッジを探す

       Args:
           edge: 現在のエッジ
           network: ネットワーク
           visited: 訪問済みエッジ

       Returns:
           隣接エッジまたはNone
       """
       try:
           node1, node2 = edge

           # node2から出ているエッジを調べる
           for neighbor in network.neighbors(node2):
               candidate_edge = (node2, neighbor)
               reverse_edge = (neighbor, node2)

               # SIM102修正: ネストしたif文を単一のif文に結合
               if (candidate_edge not in visited and reverse_edge not in visited and
                   self._check_collinearity(edge, candidate_edge, network)):
                   return candidate_edge

           return None

       except Exception:
           return None

   def _check_collinearity(self, edge1: tuple, edge2: tuple, network: nx.Graph, tolerance: float = 30.0) -> bool:
       """
       2つのエッジが共線上にあるかチェック

       Args:
           edge1, edge2: エッジ
           network: ネットワーク
           tolerance: 角度許容誤差（度）

       Returns:
           共線上にある場合True
       """
       try:
           # 座標を取得
           def get_coords(edge):
               n1, n2 = edge
               data1 = network.nodes[n1]
               data2 = network.nodes[n2]
               return (data1.get("x", 0), data1.get("y", 0)), (data2.get("x", 0), data2.get("y", 0))

           (x1, y1), (x2, y2) = get_coords(edge1)
           (x3, y3), (x4, y4) = get_coords(edge2)

           # ベクトルの角度計算
           import math

           # edge1のベクトル
           vec1 = (x2 - x1, y2 - y1)
           # edge2のベクトル
           vec2 = (x4 - x3, y4 - y3)

           # 角度差を計算
           angle1 = math.atan2(vec1[1], vec1[0])
           angle2 = math.atan2(vec2[1], vec2[0])

           angle_diff = abs(math.degrees(angle1 - angle2))
           if angle_diff > 180:
               angle_diff = 360 - angle_diff

           return angle_diff < tolerance

       except Exception:
           return False

   def _create_fallback_axial_map(self, network: nx.Graph) -> nx.Graph:
       """
       フォールバック：各エッジを1軸線として扱う

       Args:
           network: 道路ネットワーク

       Returns:
           軸線グラフ
       """
       axial_graph = nx.Graph()

       # 各エッジを個別の軸線として追加
       for i, (u, v) in enumerate(network.edges()):
           axial_graph.add_node(i,
                              edges=[(u, v)],
                              length=self._calculate_edge_length(u, v, network))

       # 隣接関係を計算
       self._connect_axial_lines_simple(axial_graph, network)

       logger.info(f"フォールバック軸線マップ作成: {axial_graph.number_of_nodes()}軸線")
       return axial_graph

   def _calculate_group_length(self, group: list[tuple], network: nx.Graph) -> float:
       """エッジグループの総長を計算"""
       total_length = 0.0
       for edge in group:
           total_length += self._calculate_edge_length(edge[0], edge[1], network)
       return total_length

   def _calculate_edge_length(self, u, v, network: nx.Graph) -> float:
       """エッジの長さを計算"""
       try:
           u_data = network.nodes[u]
           v_data = network.nodes[v]

           if "x" in u_data and "y" in u_data and "x" in v_data and "y" in v_data:
               from math import sqrt
               dx = v_data["x"] - u_data["x"]
               dy = v_data["y"] - u_data["y"]
               return sqrt(dx*dx + dy*dy)
           else:
               return 1.0  # デフォルト長
       except Exception:
           return 1.0

   def _connect_axial_lines(self, axial_graph: nx.Graph, network: nx.Graph):
       """軸線間の接続を計算"""
       nodes = list(axial_graph.nodes())

       for i in range(len(nodes)):
           for j in range(i + 1, len(nodes)):
               if self._axial_lines_intersect(nodes[i], nodes[j], axial_graph, network):
                   axial_graph.add_edge(nodes[i], nodes[j])

   def _connect_axial_lines_simple(self, axial_graph: nx.Graph, network: nx.Graph):
       """軸線間の簡単な接続計算"""
       # 共有ノードを持つエッジ同士を接続
       node_to_axial = {}

       for axial_id, data in axial_graph.nodes(data=True):
           for edge in data.get("edges", []):
               for node in edge:
                   if node not in node_to_axial:
                       node_to_axial[node] = []
                   node_to_axial[node].append(axial_id)

       # B007修正: 使用されていないloop変数をアンダースコアに変更
       # 同じノードを共有する軸線同士を接続
       for _node, axial_ids in node_to_axial.items():
           for i in range(len(axial_ids)):
               for j in range(i + 1, len(axial_ids)):
                   if not axial_graph.has_edge(axial_ids[i], axial_ids[j]):
                       axial_graph.add_edge(axial_ids[i], axial_ids[j])

   def _axial_lines_intersect(self, axial1: int, axial2: int,
                             axial_graph: nx.Graph, network: nx.Graph) -> bool:
       """2つの軸線が交差するかチェック"""
       try:
           edges1 = axial_graph.nodes[axial1].get("edges", [])
           edges2 = axial_graph.nodes[axial2].get("edges", [])

           # 共有ノードがあるかチェック
           nodes1 = set()
           for edge in edges1:
               nodes1.update(edge)

           nodes2 = set()
           for edge in edges2:
               nodes2.update(edge)

           return len(nodes1.intersection(nodes2)) > 0

       except Exception:
           return False

   def _create_convex_spaces(self, network: nx.Graph) -> list[Polygon]:
       """
       道路ネットワークから凸空間を作成

       Args:
           network: 道路ネットワーク

       Returns:
           凸空間のリスト
       """
       try:
           # 簡略化実装：ネットワークの境界を基に矩形空間を作成
           bounds = self._get_network_bounds(network)
           if not bounds:
               return []

           min_x, min_y, max_x, max_y = bounds

           # 基本的な矩形空間として返す
           from shapely.geometry import box
           return [box(min_x, min_y, max_x, max_y)]

       except Exception as e:
           logger.warning(f"凸空間作成エラー: {e}")
           return []

   def _extract_axial_lines(self, convex_spaces: list[Polygon]) -> list[LineString]:
       """
       凸空間から軸線を抽出

       Args:
           convex_spaces: 凸空間のリスト

       Returns:
           軸線のリスト
       """
       try:
           axial_lines = []

           for space in convex_spaces:
               # 凸空間の境界から軸線を生成（簡略化実装）
               bounds = space.bounds
               if len(bounds) == 4:
                   min_x, min_y, max_x, max_y = bounds

                   # 対角線を軸線として追加
                   diagonal1 = LineString([(min_x, min_y), (max_x, max_y)])
                   diagonal2 = LineString([(min_x, max_y), (max_x, min_y)])

                   axial_lines.extend([diagonal1, diagonal2])

           return axial_lines

       except Exception as e:
           logger.warning(f"軸線抽出エラー: {e}")
           return []

   def _build_axial_graph(self, axial_lines: list[LineString]) -> nx.Graph:
       """
       軸線リストからグラフを構築

       Args:
           axial_lines: 軸線のリスト

       Returns:
           軸線グラフ
       """
       try:
           axial_graph = nx.Graph()

           # 軸線をノードとして追加
           for i, line in enumerate(axial_lines):
               axial_graph.add_node(i, geometry=line, length=line.length)

           # 軸線間の交差をエッジとして追加
           for i in range(len(axial_lines)):
               for j in range(i + 1, len(axial_lines)):
                   if axial_lines[i].intersects(axial_lines[j]):
                       axial_graph.add_edge(i, j)

           return axial_graph

       except Exception as e:
           logger.warning(f"軸線グラフ構築エラー: {e}")
           return nx.Graph()

   def calculate_integration_value(self, axial_graph: nx.Graph,
                                 radius: int | None = None) -> dict[int, float]:
       """
       Integration Valueを計算

       Args:
           axial_graph: 軸線グラフ
           radius: 分析半径（Noneの場合はグローバル）

       Returns:
           各軸線のIntegration Value
       """
       try:
           if not axial_graph or axial_graph.number_of_nodes() == 0:
               return {}

           integration_values = {}

           for node in axial_graph.nodes():
               # Total Depthを計算
               if radius is None:
                   # グローバル分析
                   distances = nx.single_source_shortest_path_length(axial_graph, node)
               else:
                   # ローカル分析
                   distances = nx.single_source_shortest_path_length(
                       axial_graph, node, cutoff=radius
                   )

               if distances:
                   total_depth = sum(distances.values())
                   node_count = len(distances)

                   if node_count > 1:
                       # Mean Depth
                       mean_depth = total_depth / (node_count - 1)

                       # Integration Value (逆数)
                       integration_value = 1.0 / mean_depth if mean_depth > 0 else 0.0
                   else:
                       integration_value = 0.0
               else:
                   integration_value = 0.0

               integration_values[node] = integration_value

           return integration_values

       except Exception as e:
           logger.warning(f"Integration Value計算エラー: {e}")
           return {}

   def calculate_axial_network_metrics(self, axial_graph: nx.Graph) -> dict[str, float]:
       """
       軸線ネットワークの形態指標を計算

       Args:
           axial_graph: 軸線グラフ

       Returns:
           形態指標の辞書
       """
       try:
           if not axial_graph or axial_graph.number_of_nodes() == 0:
               return self._empty_axial_metrics()

           num_nodes = axial_graph.number_of_nodes()
           num_edges = axial_graph.number_of_edges()

           # 基本指標
           metrics = {
               "axial_lines": num_nodes,
               "axial_connections": num_edges,
               "axial_islands": num_nodes - nx.number_connected_components(axial_graph),
           }

           # Grid Axiality (GA)
           if num_nodes > 0:
               max_possible_edges = num_nodes * (num_nodes - 1) // 2
               metrics["grid_axiality"] = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
           else:
               metrics["grid_axiality"] = 0.0

           # Axial Ringiness (AR)
           cycles = num_edges - num_nodes + nx.number_connected_components(axial_graph)
           max_cycles = (num_nodes - nx.number_connected_components(axial_graph)) if num_nodes > 0 else 0
           metrics["axial_ringiness"] = cycles / max_cycles if max_cycles > 0 else 0.0

           # Axial Articulation (AA)
           articulation_points = len(list(nx.articulation_points(axial_graph)))
           metrics["axial_articulation"] = articulation_points / num_nodes if num_nodes > 0 else 0.0

           return metrics

       except Exception as e:
           logger.warning(f"軸線ネットワーク指標計算エラー: {e}")
           return self._empty_axial_metrics()

   def _get_network_bounds(self, network: nx.Graph) -> tuple[float, float, float, float] | None:
       """
       ネットワークの境界を取得

       Args:
           network: ネットワークグラフ

       Returns:
           境界 (min_x, min_y, max_x, max_y) またはNone
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
       軸線分析の包括的実行（デバッグ強化版）

       Args:
           network: 道路ネットワーク

       Returns:
           軸線分析の全結果
       """
       try:
           logger.info(f"軸線分析開始: ネットワーク {network.number_of_nodes()}ノード, {network.number_of_edges()}エッジ")

           # 軸線マップ作成
           axial_map = self.create_axial_map(network)
           logger.info(f"軸線マップ作成結果: {axial_map.number_of_nodes()}軸線, {axial_map.number_of_edges()}接続")

           if axial_map.number_of_nodes() == 0:
               logger.warning("軸線が生成されませんでした")
               return self._empty_axial_result()

           # Integration Value計算
           global_integration = self.analyze_global_integration(axial_map)
           local_integration = self.analyze_local_integration(axial_map, radius=3)

           logger.info(f"Integration計算完了: global={len(global_integration)}, local={len(local_integration)}")

           # 形態指標計算
           network_metrics = self.calculate_axial_network_metrics(axial_map)
           logger.info(f"形態指標: {network_metrics}")

           # 統計計算
           integration_stats = self._calculate_integration_statistics(global_integration)
           logger.info(f"統計: {integration_stats}")

           return {
               "axial_map": axial_map,
               "global_integration": global_integration,
               "local_integration": local_integration,
               "network_metrics": network_metrics,
               "integration_statistics": integration_stats,
           }

       except Exception as e:
           logger.error(f"軸線分析エラー: {e}")
           import traceback
           traceback.print_exc()
           return self._empty_axial_result()

   def _empty_axial_result(self) -> dict[str, Any]:
       """空の軸線分析結果を返す"""
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
