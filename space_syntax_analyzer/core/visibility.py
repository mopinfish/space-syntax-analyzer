"""
可視領域分析モジュール - VisibilityAnalyzer

Isovist分析とVisibility Graph Analysis（VGA）を実装します。
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull, distance
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class VisibilityAnalyzer:
    """
    可視領域分析を行うクラス

    Isovist分析とVGAの機能を提供します。
    """

    def __init__(self,
                 visibility_radius: float = 100.0,
                 angular_resolution: int = 360,
                 building_height: float = 3.0) -> None:
        """
        VisibilityAnalyzerを初期化

        Args:
            visibility_radius: 可視範囲（メートル）
            angular_resolution: 角度分解能
            building_height: 建物高さ（メートル）
        """
        self.visibility_radius = visibility_radius
        self.angular_resolution = angular_resolution
        self.building_height = building_height

    def calculate_isovist(self, observation_point: tuple[float, float],
                         network: nx.Graph,
                         obstacles: list[Polygon] | None = None) -> dict[str, Any]:
        """
        指定点からのIsovistを計算

        Args:
            observation_point: 観測点座標 (x, y)
            network: 道路ネットワーク
            obstacles: 障害物ポリゴンのリスト

        Returns:
            Isovist分析結果
        """
        try:
            logger.info(f"Isovist計算開始: 観測点{observation_point}")

            # 観測点の準備
            obs_point = Point(observation_point)

            # 障害物の準備
            if obstacles is None:
                obstacles = self._extract_obstacles_from_network(network)

            # 可視境界の計算
            visibility_polygon = self._calculate_visibility_polygon(
                obs_point, obstacles
            )

            # Isovist指標の計算
            isovist_metrics = self._calculate_isovist_metrics(
                visibility_polygon, obs_point
            )

            return {
                "observation_point": observation_point,
                "visibility_polygon": visibility_polygon,
                "obstacles": obstacles,
                **isovist_metrics
            }

        except Exception as e:
            logger.error(f"Isovist計算エラー: {e}")
            return self._empty_isovist_result(observation_point)

    def _extract_obstacles_from_network(self, network: nx.Graph) -> list[Polygon]:
        """
        ネットワークから障害物を抽出

        Args:
            network: 道路ネットワーク

        Returns:
            障害物ポリゴンのリスト
        """
        try:
            obstacles = []

            # エッジを線分として障害物に追加
            for u, v, data in network.edges(data=True):
                u_data = network.nodes[u]
                v_data = network.nodes[v]

                if all(key in u_data for key in ["x", "y"]) and \
                   all(key in v_data for key in ["x", "y"]):

                    # 道路幅を考慮した障害物作成
                    line = LineString([(u_data["x"], u_data["y"]),
                                     (v_data["x"], v_data["y"])])

                    # 建物を道路沿いに配置（簡易版）
                    # 実際の実装では、建物データを別途取得する必要がある
                    building_width = data.get("width", 4.0)
                    building_polygon = line.buffer(building_width / 2)

                    if isinstance(building_polygon, Polygon):
                        obstacles.append(building_polygon)

            return obstacles

        except Exception as e:
            logger.warning(f"障害物抽出エラー: {e}")
            return []

    def _calculate_visibility_polygon(self, observation_point: Point,
                                    obstacles: list[Polygon]) -> Polygon:
        """
        可視領域ポリゴンを計算

        Args:
            observation_point: 観測点
            obstacles: 障害物リスト

        Returns:
            可視領域ポリゴン
        """
        try:
            # 観測点を中心とした円形領域
            base_circle = observation_point.buffer(self.visibility_radius)

            if not obstacles:
                return base_circle

            # 障害物による陰影の計算
            shadow_polygons = []

            for obstacle in obstacles:
                if not obstacle.is_valid or obstacle.is_empty:
                    continue

                # 障害物と観測点の関係をチェック
                if observation_point.distance(obstacle) > self.visibility_radius:
                    continue

                # 陰影ポリゴンの計算
                shadow = self._calculate_shadow(observation_point, obstacle)
                if shadow and shadow.is_valid:
                    shadow_polygons.append(shadow)

            # 可視領域 = 基本円 - 全陰影
            if shadow_polygons:
                all_shadows = unary_union(shadow_polygons)
                visible_area = base_circle.difference(all_shadows)

                if isinstance(visible_area, Polygon):
                    return visible_area
                else:
                    # MultiPolygonの場合は最大面積のポリゴンを選択
                    if hasattr(visible_area, "geoms"):
                        largest = max(visible_area.geoms, key=lambda x: x.area)
                        return largest

            return base_circle

        except Exception as e:
            logger.warning(f"可視領域計算エラー: {e}")
            return observation_point.buffer(self.visibility_radius)

    def _calculate_shadow(self, observation_point: Point,
                         obstacle: Polygon) -> Polygon | None:
        """
        障害物による陰影を計算

        Args:
            observation_point: 観測点
            obstacle: 障害物

        Returns:
            陰影ポリゴン
        """
        try:
            obs_x, obs_y = observation_point.x, observation_point.y

            # 障害物の境界点を取得
            obstacle_coords = list(obstacle.exterior.coords)

            # 陰影計算用の頂点を生成
            shadow_points = []

            for coord in obstacle_coords:
                if len(coord) >= 2:
                    obj_x, obj_y = coord[0], coord[1]

                    # 観測点から障害物頂点への方向ベクトル
                    dx = obj_x - obs_x
                    dy = obj_y - obs_y

                    # 陰影の延長点を計算
                    distance_to_obstacle = np.sqrt(dx**2 + dy**2)
                    if distance_to_obstacle > 0:
                        # 可視範囲まで延長
                        extension_factor = self.visibility_radius / distance_to_obstacle
                        shadow_x = obs_x + dx * extension_factor
                        shadow_y = obs_y + dy * extension_factor
                        shadow_points.append((shadow_x, shadow_y))

            if len(shadow_points) >= 3:
                # 障害物の点と陰影の点を結合してポリゴンを作成
                all_points = obstacle_coords + shadow_points

                # 凸包を計算
                try:
                    hull = ConvexHull(all_points)
                    hull_points = [all_points[i] for i in hull.vertices]
                    return Polygon(hull_points)
                except (ValueError, IndexError) as e:
                    # 凸包計算に失敗した場合は障害物自体を返す
                    logger.warning(f"凸包計算エラー: {e}")
                    return obstacle

            return None

        except Exception as e:
            logger.warning(f"陰影計算エラー: {e}")
            return None

    def _calculate_isovist_metrics(self, visibility_polygon: Polygon,
                                  observation_point: Point) -> dict[str, float]:
        """
        Isovist指標を計算

        Args:
            visibility_polygon: 可視領域ポリゴン
            observation_point: 観測点

        Returns:
            Isovist指標
        """
        try:
            # 基本指標
            area = visibility_polygon.area
            perimeter = visibility_polygon.length

            # 形状指標
            compactness = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0

            # 重心との距離
            centroid = visibility_polygon.centroid
            centroid_distance = observation_point.distance(centroid)

            # 最大可視距離
            boundary_coords = list(visibility_polygon.exterior.coords)
            max_distance = 0.0

            for coord in boundary_coords:
                if len(coord) >= 2:
                    boundary_point = Point(coord[0], coord[1])
                    dist = observation_point.distance(boundary_point)
                    max_distance = max(max_distance, dist)

            # 最小可視距離
            min_distance = min(
                observation_point.distance(Point(coord[0], coord[1]))
                for coord in boundary_coords
                if len(coord) >= 2
            ) if boundary_coords else 0.0

            # Drift（重心偏移）
            drift = centroid_distance

            # Occlusivity（遮蔽性）
            max_possible_area = np.pi * (self.visibility_radius ** 2)
            occlusivity = 1.0 - (area / max_possible_area) if max_possible_area > 0 else 0.0

            return {
                "visible_area": area,
                "perimeter": perimeter,
                "compactness": compactness,
                "max_visible_distance": max_distance,
                "min_visible_distance": min_distance,
                "drift": drift,
                "occlusivity": occlusivity,
            }

        except Exception as e:
            logger.warning(f"Isovist指標計算エラー: {e}")
            return {
                "visible_area": 0.0,
                "perimeter": 0.0,
                "compactness": 0.0,
                "max_visible_distance": 0.0,
                "min_visible_distance": 0.0,
                "drift": 0.0,
                "occlusivity": 1.0,
            }

    def analyze_visibility_field(self, network: nx.Graph,
                               sampling_distance: float = 50.0) -> dict[str, Any]:
        """
        道路ネットワーク全体の可視領域分析

        Args:
            network: 道路ネットワーク
            sampling_distance: サンプリング間隔（メートル）

        Returns:
            可視領域分析結果
        """
        try:
            logger.info("可視領域フィールド分析開始")

            # サンプリング点の生成
            sampling_points = self._generate_sampling_points(network, sampling_distance)

            if not sampling_points:
                logger.warning("サンプリング点が生成されませんでした")
                return self._empty_visibility_field_result()

            # 各点でのIsovist計算
            isovist_results = []
            obstacles = self._extract_obstacles_from_network(network)

            for i, point in enumerate(sampling_points):
                if i % 10 == 0:  # 進捗表示
                    logger.info(f"Isovist計算進捗: {i}/{len(sampling_points)}")

                isovist = self.calculate_isovist(point, network, obstacles)
                isovist_results.append(isovist)

            # 統計分析
            field_statistics = self._calculate_field_statistics(isovist_results)

            # 可視領域の変動性分析
            variability_metrics = self._calculate_visibility_variability(isovist_results)

            return {
                "sampling_points": sampling_points,
                "isovist_results": isovist_results,
                "field_statistics": field_statistics,
                "variability_metrics": variability_metrics,
            }

        except Exception as e:
            logger.error(f"可視領域フィールド分析エラー: {e}")
            return self._empty_visibility_field_result()

    def _generate_sampling_points(self, network: nx.Graph,
                                sampling_distance: float) -> list[tuple[float, float]]:
        """
        道路ネットワーク上にサンプリング点を生成

        Args:
            network: 道路ネットワーク
            sampling_distance: サンプリング間隔

        Returns:
            サンプリング点のリスト
        """
        try:
            sampling_points = []

            for u, v, _data in network.edges(data=True):
                u_data = network.nodes[u]
                v_data = network.nodes[v]

                if all(key in u_data for key in ["x", "y"]) and \
                   all(key in v_data for key in ["x", "y"]):

                    start_point = (u_data["x"], u_data["y"])
                    end_point = (v_data["x"], v_data["y"])

                    # エッジ上にサンプリング点を配置
                    edge_points = self._sample_points_on_edge(
                        start_point, end_point, sampling_distance
                    )
                    sampling_points.extend(edge_points)

            # 重複点の除去
            unique_points = []
            for point in sampling_points:
                is_duplicate = False
                for existing_point in unique_points:
                    if distance.euclidean(point, existing_point) < sampling_distance / 2:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)

            logger.info(f"サンプリング点生成完了: {len(unique_points)}点")
            return unique_points

        except Exception as e:
            logger.warning(f"サンプリング点生成エラー: {e}")
            return []

    def _sample_points_on_edge(self, start: tuple[float, float],
                              end: tuple[float, float],
                              interval: float) -> list[tuple[float, float]]:
        """
        エッジ上にサンプリング点を配置

        Args:
            start: 開始点
            end: 終了点
            interval: サンプリング間隔

        Returns:
            サンプリング点のリスト
        """
        try:
            edge_length = distance.euclidean(start, end)

            if edge_length <= interval:
                return [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]

            num_points = int(edge_length / interval)
            points = []

            for i in range(1, num_points):
                ratio = i / num_points
                x = start[0] + (end[0] - start[0]) * ratio
                y = start[1] + (end[1] - start[1]) * ratio
                points.append((x, y))

            return points

        except Exception:
            return []

    def _calculate_field_statistics(self, isovist_results: list[dict[str, Any]]) -> dict[str, float]:
        """
        可視領域フィールドの統計を計算

        Args:
            isovist_results: Isovist結果のリスト

        Returns:
            フィールド統計
        """
        try:
            if not isovist_results:
                return {}

            # 可視面積の統計
            areas = [result.get("visible_area", 0) for result in isovist_results]

            # コンパクト性の統計
            compactness_values = [result.get("compactness", 0) for result in isovist_results]

            # 遮蔽性の統計
            occlusivity_values = [result.get("occlusivity", 0) for result in isovist_results]

            return {
                "mean_visible_area": float(np.mean(areas)),
                "std_visible_area": float(np.std(areas)),
                "min_visible_area": float(np.min(areas)),
                "max_visible_area": float(np.max(areas)),
                "mean_compactness": float(np.mean(compactness_values)),
                "std_compactness": float(np.std(compactness_values)),
                "mean_occlusivity": float(np.mean(occlusivity_values)),
                "std_occlusivity": float(np.std(occlusivity_values)),
                "total_sampling_points": len(isovist_results),
            }

        except Exception as e:
            logger.warning(f"フィールド統計計算エラー: {e}")
            return {}

    def _calculate_visibility_variability(self, isovist_results: list[dict[str, Any]]) -> dict[str, float]:
        """
        可視領域の変動性を分析

        Args:
            isovist_results: Isovist結果のリスト

        Returns:
            変動性指標
        """
        try:
            if not isovist_results:
                return {}

            # 可視面積の変動係数
            areas = [result.get("visible_area", 0) for result in isovist_results]
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0

            # コンパクト性の変動係数
            compactness_values = [result.get("compactness", 0) for result in isovist_results]
            compactness_cv = (np.std(compactness_values) / np.mean(compactness_values)
                            if np.mean(compactness_values) > 0 else 0)

            # 空間多様性指標（エントロピー）
            diversity_index = self._calculate_spatial_diversity(areas)

            return {
                "area_coefficient_variation": float(area_cv),
                "compactness_coefficient_variation": float(compactness_cv),
                "spatial_diversity_index": float(diversity_index),
            }

        except Exception as e:
            logger.warning(f"変動性計算エラー: {e}")
            return {}

    def _calculate_spatial_diversity(self, values: list[float]) -> float:
        """
        空間多様性指標（シャノンエントロピー）を計算

        Args:
            values: 分析対象の値リスト

        Returns:
            多様性指標
        """
        try:
            if not values or all(v == 0 for v in values):
                return 0.0

            # 値を正規化してカテゴリに分割
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                return 0.0

            # 10段階のカテゴリに分割
            num_bins = 10
            bin_size = (max_val - min_val) / num_bins
            bins = [0] * num_bins

            for value in values:
                if value >= max_val:
                    bin_index = num_bins - 1
                else:
                    bin_index = int((value - min_val) / bin_size)
                    bin_index = max(0, min(bin_index, num_bins - 1))
                bins[bin_index] += 1

            # シャノンエントロピーの計算
            total = sum(bins)
            if total == 0:
                return 0.0

            entropy = 0.0
            for count in bins:
                if count > 0:
                    probability = count / total
                    entropy -= probability * np.log2(probability)

            return entropy

        except Exception as e:
            logger.warning(f"多様性指標計算エラー: {e}")
            return 0.0

    def analyze_visual_connectivity(self, network: nx.Graph) -> dict[str, Any]:
        """
        視覚的接続性の分析

        Args:
            network: 道路ネットワーク

        Returns:
            視覚的接続性分析結果
        """
        try:
            logger.info("視覚的接続性分析開始")

            # 主要交差点の抽出
            major_intersections = self._extract_major_intersections(network)

            if not major_intersections:
                logger.warning("主要交差点が見つかりませんでした")
                return self._empty_visual_connectivity_result()

            # 交差点間の視覚的接続性を計算
            visual_connections = {}

            for i, intersection1 in enumerate(major_intersections):
                for j, intersection2 in enumerate(major_intersections[i+1:], i+1):
                    connectivity = self._calculate_visual_connection(
                        intersection1, intersection2, network
                    )
                    visual_connections[(i, j)] = connectivity

            # 視覚的接続性ネットワークの構築
            visual_network = self._build_visual_network(
                major_intersections, visual_connections
            )

            # ネットワーク指標の計算
            network_metrics = self._calculate_visual_network_metrics(visual_network)

            return {
                "major_intersections": major_intersections,
                "visual_connections": visual_connections,
                "visual_network": visual_network,
                "network_metrics": network_metrics,
            }

        except Exception as e:
            logger.error(f"視覚的接続性分析エラー: {e}")
            return self._empty_visual_connectivity_result()

    def _extract_major_intersections(self, network: nx.Graph) -> list[tuple[float, float]]:
        """
        主要交差点を抽出

        Args:
            network: 道路ネットワーク

        Returns:
            主要交差点の座標リスト
        """
        intersections = []

        for node, data in network.nodes(data=True):
            # 次数3以上を交差点とする
            if network.degree[node] >= 3 and "x" in data and "y" in data:
                intersections.append((data["x"], data["y"]))

        return intersections

    def _calculate_visual_connection(self, point1: tuple[float, float],
                                   point2: tuple[float, float],
                                   network: nx.Graph) -> float:
        """
        2点間の視覚的接続性を計算

        Args:
            point1, point2: 分析対象の2点
            network: 道路ネットワーク

        Returns:
            視覚的接続性スコア（0-1）
        """
        try:
            # 2点間の直線
            sight_line = LineString([point1, point2])

            # 障害物との干渉チェック
            obstacles = self._extract_obstacles_from_network(network)

            total_obstruction = 0.0
            for obstacle in obstacles:
                if sight_line.intersects(obstacle):
                    intersection = sight_line.intersection(obstacle)
                    if hasattr(intersection, "length"):
                        total_obstruction += intersection.length

            # 視覚的接続性スコア
            sight_line_length = sight_line.length
            if sight_line_length > 0:
                obstruction_ratio = total_obstruction / sight_line_length
                connectivity_score = max(0.0, 1.0 - obstruction_ratio)
            else:
                connectivity_score = 0.0

            return connectivity_score

        except Exception as e:
            logger.warning(f"視覚的接続性計算エラー: {e}")
            return 0.0

    def _build_visual_network(self, intersections: list[tuple[float, float]],
                            connections: dict[tuple[int, int], float]) -> nx.Graph:
        """
        視覚的接続性ネットワークを構築

        Args:
            intersections: 交差点リスト
            connections: 接続性スコア

        Returns:
            視覚的接続性ネットワーク
        """
        graph = nx.Graph()

        # ノードの追加
        for i, (x, y) in enumerate(intersections):
            graph.add_node(i, x=x, y=y)

        # エッジの追加（閾値以上の接続性を持つもの）
        connection_threshold = 0.5

        for (i, j), score in connections.items():
            if score >= connection_threshold:
                graph.add_edge(i, j, weight=score, visual_connectivity=score)

        return graph

    def _calculate_visual_network_metrics(self, visual_network: nx.Graph) -> dict[str, float]:
        """
        視覚的接続性ネットワークの指標を計算

        Args:
            visual_network: 視覚的接続性ネットワーク

        Returns:
            ネットワーク指標
        """
        try:
            if visual_network.number_of_nodes() == 0:
                return {}

            # 基本統計
            num_nodes = visual_network.number_of_nodes()
            num_edges = visual_network.number_of_edges()

            # 密度
            density = nx.density(visual_network)

            # クラスタリング係数
            clustering = nx.average_clustering(visual_network)

            # 平均パス長（連結グラフの場合のみ）
            if nx.is_connected(visual_network):
                avg_path_length = nx.average_shortest_path_length(visual_network)
            else:
                # 最大連結成分での計算
                largest_cc = max(nx.connected_components(visual_network), key=len)
                subgraph = visual_network.subgraph(largest_cc)
                if subgraph.number_of_nodes() > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 0.0

            # 平均視覚的接続性
            if num_edges > 0:
                connection_scores = [data["visual_connectivity"]
                                   for _, _, data in visual_network.edges(data=True)]
                avg_visual_connectivity = np.mean(connection_scores)
            else:
                avg_visual_connectivity = 0.0

            return {
                "visual_nodes": num_nodes,
                "visual_edges": num_edges,
                "visual_density": density,
                "visual_clustering": clustering,
                "avg_visual_path_length": avg_path_length,
                "avg_visual_connectivity": avg_visual_connectivity,
            }

        except Exception as e:
            logger.warning(f"視覚的ネットワーク指標計算エラー: {e}")
            return {}

    def calculate_visibility_graph(self, network: nx.Graph) -> nx.Graph:
        """
        Visibility Graph Analysis（VGA）を実行

        Args:
            network: 道路ネットワーク

        Returns:
            可視グラフ
        """
        try:
            logger.info("Visibility Graph Analysis開始")

            # グリッドベースのサンプリング点生成
            grid_points = self._generate_grid_points(network, grid_size=25.0)

            # 可視グラフの構築
            visibility_graph = nx.Graph()

            # ノードの追加
            for i, (x, y) in enumerate(grid_points):
                visibility_graph.add_node(i, x=x, y=y)

            # 相互可視性チェック
            obstacles = self._extract_obstacles_from_network(network)

            for i in range(len(grid_points)):
                for j in range(i + 1, len(grid_points)):
                    if self._is_mutually_visible(grid_points[i], grid_points[j], obstacles):
                        dist = distance.euclidean(grid_points[i], grid_points[j])
                        visibility_graph.add_edge(i, j, weight=dist)

            logger.info(f"VGA完了: {visibility_graph.number_of_nodes()}ノード, "
                       f"{visibility_graph.number_of_edges()}エッジ")

            return visibility_graph

        except Exception as e:
            logger.error(f"VGA実行エラー: {e}")
            return nx.Graph()

    def _generate_grid_points(self, network: nx.Graph,
                            grid_size: float = 25.0) -> list[tuple[float, float]]:
        """
        グリッドベースのサンプリング点を生成

        Args:
            network: 道路ネットワーク
            grid_size: グリッドサイズ（メートル）

        Returns:
            グリッド点のリスト
        """
        try:
            bounds = self._get_network_bounds(network)
            if not bounds:
                return []

            min_x, min_y, max_x, max_y = bounds

            grid_points = []
            current_x = min_x

            while current_x <= max_x:
                current_y = min_y
                while current_y <= max_y:
                    grid_points.append((current_x, current_y))
                    current_y += grid_size
                current_x += grid_size

            return grid_points

        except Exception as e:
            logger.warning(f"グリッド点生成エラー: {e}")
            return []

    def _is_mutually_visible(self, point1: tuple[float, float],
                           point2: tuple[float, float],
                           obstacles: list[Polygon]) -> bool:
        """
        2点間が相互に可視かどうかを判定

        Args:
            point1, point2: 判定対象の2点
            obstacles: 障害物リスト

        Returns:
            相互可視かどうか
        """
        try:
            sight_line = LineString([point1, point2])

            # 距離チェック
            if sight_line.length > self.visibility_radius:
                return False

            # 障害物との干渉チェック
            for obstacle in obstacles:
                if sight_line.intersects(obstacle):
                    # 接触のみの場合は可視とする
                    intersection = sight_line.intersection(obstacle)
                    if hasattr(intersection, "length") and intersection.length > 1.0:
                        return False

            return True

        except Exception:
            return False

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

    def _empty_isovist_result(self, observation_point: tuple[float, float]) -> dict[str, Any]:
        """空のIsovist結果を返す"""
        return {
            "observation_point": observation_point,
            "visibility_polygon": Point(observation_point).buffer(0),
            "obstacles": [],
            "visible_area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "max_visible_distance": 0.0,
            "min_visible_distance": 0.0,
            "drift": 0.0,
            "occlusivity": 1.0,
        }

    def _empty_visibility_field_result(self) -> dict[str, Any]:
        """空の可視領域フィールド結果を返す"""
        return {
            "sampling_points": [],
            "isovist_results": [],
            "field_statistics": {},
            "variability_metrics": {},
        }

    def _empty_visual_connectivity_result(self) -> dict[str, Any]:
        """空の視覚的接続性結果を返す"""
        return {
            "major_intersections": [],
            "visual_connections": {},
            "visual_network": nx.Graph(),
            "network_metrics": {},
        }

    def export_visibility_results(self, results: dict[str, Any],
                                output_path: str,
                                format_type: str = "geojson") -> None:
        """
        可視領域分析結果をエクスポート

        Args:
            results: 分析結果
            output_path: 出力パス
            format_type: 出力形式
        """
        try:
            if format_type.lower() == "geojson":
                self._export_visibility_geojson(results, output_path)
            elif format_type.lower() == "csv":
                self._export_visibility_csv(results, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"可視領域結果エクスポートエラー: {e}")
            raise

    def _export_visibility_geojson(self, results: dict[str, Any], output_path: str) -> None:
        """可視領域結果をGeoJSON形式でエクスポート"""
        try:
            import json

            from shapely.geometry import mapping

            features = []

            # サンプリング点の追加
            for i, point in enumerate(results.get("sampling_points", [])):
                feature = {
                    "type": "Feature",
                    "geometry": mapping(Point(point)),
                    "properties": {
                        "type": "sampling_point",
                        "point_id": i
                    }
                }
                features.append(feature)

            # 可視領域ポリゴンの追加
            for i, isovist in enumerate(results.get("isovist_results", [])):
                if "visibility_polygon" in isovist:
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(isovist["visibility_polygon"]),
                        "properties": {
                            "type": "visibility_polygon",
                            "point_id": i,
                            "visible_area": isovist.get("visible_area", 0),
                            "compactness": isovist.get("compactness", 0),
                        }
                    }
                    features.append(feature)

            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"GeoJSONエクスポートエラー: {e}")
            raise

    def _export_visibility_csv(self, results: dict[str, Any], output_path: str) -> None:
        """可視領域結果をCSV形式でエクスポート"""
        try:
            import pandas as pd

            data = []

            for i, isovist in enumerate(results.get("isovist_results", [])):
                row = {
                    "point_id": i,
                    "x": isovist.get("observation_point", [0, 0])[0],
                    "y": isovist.get("observation_point", [0, 0])[1],
                    "visible_area": isovist.get("visible_area", 0),
                    "perimeter": isovist.get("perimeter", 0),
                    "compactness": isovist.get("compactness", 0),
                    "max_visible_distance": isovist.get("max_visible_distance", 0),
                    "min_visible_distance": isovist.get("min_visible_distance", 0),
                    "drift": isovist.get("drift", 0),
                    "occlusivity": isovist.get("occlusivity", 0),
                }
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

        except Exception as e:
            logger.error(f"CSVエクスポートエラー: {e}")
            raise
