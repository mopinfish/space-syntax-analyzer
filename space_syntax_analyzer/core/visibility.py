# space_syntax_analyzer/core/visibility_fixed.py
"""
可視領域分析モジュール - VisibilityAnalyzer（エラー修正版）

Isovist分析とVisibility Graph Analysis（VGA）を実装します。
データ型エラーを修正し、堅牢性を向上させました。
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
        ネットワークから障害物を抽出（データ型エラー修正版）

        Args:
            network: 道路ネットワーク

        Returns:
            障害物ポリゴンのリスト
        """
        try:
            obstacles = []

            # エッジを線分として障害物に追加
            for u, v, data in network.edges(data=True):
                try:
                    u_data = network.nodes[u]
                    v_data = network.nodes[v]

                    if all(key in u_data for key in ["x", "y"]) and \
                       all(key in v_data for key in ["x", "y"]):

                        # 道路幅を取得（データ型安全版）
                        building_width = self._safe_get_width(data)

                        # 道路を線分として作成
                        line = LineString([(u_data["x"], u_data["y"]),
                                         (v_data["x"], v_data["y"])])

                        # 建物を道路沿いに配置（簡易版）
                        building_polygon = line.buffer(building_width / 2)

                        if isinstance(building_polygon, Polygon) and building_polygon.is_valid:
                            obstacles.append(building_polygon)

                except Exception as edge_error:
                    logger.debug(f"エッジ処理エラー（スキップ）: {edge_error}")
                    continue

            logger.debug(f"障害物抽出完了: {len(obstacles)}個")
            return obstacles

        except Exception as e:
            logger.warning(f"障害物抽出エラー: {e}")
            return []

    def _safe_get_width(self, edge_data: dict) -> float:
        """
        エッジデータから安全に幅を取得

        Args:
            edge_data: エッジのデータ辞書

        Returns:
            幅の値（メートル）
        """
        try:
            # 優先順位で幅データを取得
            width_candidates = ["width", "est_width", "lanes"]

            for width_key in width_candidates:
                if width_key in edge_data:
                    width_value = edge_data[width_key]

                    # データ型に応じて処理
                    if isinstance(width_value, int | float):
                        return max(float(width_value), 1.0)  # 最小1m

                    elif isinstance(width_value, str):
                        # 文字列から数値を抽出
                        parsed_width = self._parse_width_string(width_value)
                        if parsed_width is not None:
                            return max(parsed_width, 1.0)

                    elif isinstance(width_value, list) and width_value:
                        # リストの場合は最初の値を使用
                        first_value = width_value[0]
                        if isinstance(first_value, int | float):
                            return max(float(first_value), 1.0)
                        elif isinstance(first_value, str):
                            parsed_width = self._parse_width_string(first_value)
                            if parsed_width is not None:
                                return max(parsed_width, 1.0)

            # レーン数から推定
            if "lanes" in edge_data:
                lanes = edge_data["lanes"]
                if isinstance(lanes, int | float):
                    return max(float(lanes) * 3.5, 4.0)  # 1レーン3.5m想定
                elif isinstance(lanes, str):
                    parsed_lanes = self._parse_width_string(lanes)
                    if parsed_lanes is not None:
                        return max(parsed_lanes * 3.5, 4.0)

            # 道路タイプから推定
            highway_type = edge_data.get("highway", "")
            if highway_type:
                return self._estimate_width_from_highway_type(highway_type)

            # デフォルト値
            return 4.0

        except Exception as e:
            logger.debug(f"幅取得エラー: {e}")
            return 4.0  # デフォルト値

    def _parse_width_string(self, width_str: str) -> float | None:
        """
        文字列から幅の数値を抽出

        Args:
            width_str: 幅を表す文字列

        Returns:
            抽出された数値、失敗時はNone
        """
        try:
            import re

            # 数値部分を抽出（小数点を含む）
            match = re.search(r"(\d+(?:\.\d+)?)", width_str)
            if match:
                return float(match.group(1))

            return None

        except Exception:
            return None

    def _estimate_width_from_highway_type(self, highway_type: str) -> float:
        """
        道路タイプから幅を推定

        Args:
            highway_type: OSMの道路タイプ

        Returns:
            推定幅（メートル）
        """
        # 道路タイプごとの典型的な幅
        width_map = {
            "motorway": 12.0,
            "trunk": 10.0,
            "primary": 8.0,
            "secondary": 6.0,
            "tertiary": 5.0,
            "residential": 4.0,
            "service": 3.0,
            "footway": 2.0,
            "path": 1.5,
            "cycleway": 2.0,
        }

        return width_map.get(highway_type, 4.0)

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
                try:
                    all_shadows = unary_union(shadow_polygons)
                    visible_area = base_circle.difference(all_shadows)

                    if isinstance(visible_area, Polygon) and visible_area.is_valid:
                        return visible_area
                    elif hasattr(visible_area, "geoms"):
                        # MultiPolygonの場合は最大面積のポリゴンを選択
                        valid_geoms = [geom for geom in visible_area.geoms
                                     if isinstance(geom, Polygon) and geom.is_valid]
                        if valid_geoms:
                            largest = max(valid_geoms, key=lambda x: x.area)
                            return largest
                except Exception as shadow_error:
                    logger.debug(f"陰影計算エラー: {shadow_error}")

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
                    shadow_polygon = Polygon(hull_points)

                    if shadow_polygon.is_valid:
                        return shadow_polygon
                    else:
                        # 無効なポリゴンの場合は元の障害物を返す
                        return obstacle

                except (ValueError, IndexError) as e:
                    logger.debug(f"凸包計算エラー: {e}")
                    return obstacle

            return None

        except Exception as e:
            logger.debug(f"陰影計算エラー: {e}")
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
            if not visibility_polygon.is_valid or visibility_polygon.is_empty:
                return self._get_empty_isovist_metrics()

            # 基本指標
            area = visibility_polygon.area
            perimeter = visibility_polygon.length

            # 形状指標
            compactness = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0

            # 重心との距離
            try:
                centroid = visibility_polygon.centroid
                centroid_distance = observation_point.distance(centroid)
            except Exception:
                centroid_distance = 0.0

            # 最大・最小可視距離
            boundary_coords = list(visibility_polygon.exterior.coords)
            distances = []

            for coord in boundary_coords:
                if len(coord) >= 2:
                    try:
                        boundary_point = Point(coord[0], coord[1])
                        dist = observation_point.distance(boundary_point)
                        distances.append(dist)
                    except Exception as e:
                        logger.warning(f"距離計算エラー: {e}")
                        continue

            max_distance = max(distances) if distances else 0.0
            min_distance = min(distances) if distances else 0.0

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
            return self._get_empty_isovist_metrics()

    def _get_empty_isovist_metrics(self) -> dict[str, float]:
        """空のIsovist指標を返す"""
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

            # 障害物を事前に抽出（効率化）
            obstacles = self._extract_obstacles_from_network(network)

            # 各点でのIsovist計算
            isovist_results = []
            total_points = len(sampling_points)

            for i, point in enumerate(sampling_points):
                if i % max(10, total_points // 10) == 0:  # 進捗表示
                    logger.info(f"Isovist計算進捗: {i}/{total_points}")

                try:
                    isovist = self.calculate_isovist(point, network, obstacles)
                    isovist_results.append(isovist)
                except Exception as point_error:
                    logger.debug(f"点 {i} でのIsovist計算エラー: {point_error}")
                    # エラーの場合は空の結果を追加
                    isovist_results.append(self._empty_isovist_result(point))

            # 統計分析
            field_statistics = self._calculate_field_statistics(isovist_results)

            # 可視領域の変動性分析
            variability_metrics = self._calculate_visibility_variability(isovist_results)

            logger.info(f"可視領域フィールド分析完了: {len(isovist_results)}点")

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
                try:
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

                except Exception as edge_error:
                    logger.debug(f"エッジサンプリングエラー: {edge_error}")
                    continue

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
                return [((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)]

            num_points = max(1, int(edge_length / interval))
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

            # 有効な結果のみを抽出
            valid_results = [result for result in isovist_results
                           if "visible_area" in result and isinstance(result["visible_area"], int | float)]

            if not valid_results:
                return {}

            # 各指標の値を抽出
            areas = [result["visible_area"] for result in valid_results]
            compactness_values = [result.get("compactness", 0) for result in valid_results]
            occlusivity_values = [result.get("occlusivity", 0) for result in valid_results]

            return {
                "mean_visible_area": float(np.mean(areas)),
                "std_visible_area": float(np.std(areas)),
                "min_visible_area": float(np.min(areas)),
                "max_visible_area": float(np.max(areas)),
                "mean_compactness": float(np.mean(compactness_values)),
                "std_compactness": float(np.std(compactness_values)),
                "mean_occlusivity": float(np.mean(occlusivity_values)),
                "std_occlusivity": float(np.std(occlusivity_values)),
                "total_sampling_points": len(valid_results),
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

            # 有効な結果のみを抽出
            valid_results = [result for result in isovist_results
                           if "visible_area" in result and isinstance(result["visible_area"], int | float)]

            if not valid_results:
                return {}

            # 可視面積の変動係数
            areas = [result["visible_area"] for result in valid_results]
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0

            # コンパクト性の変動係数
            compactness_values = [result.get("compactness", 0) for result in valid_results]
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
                    try:
                        connectivity = self._calculate_visual_connection(
                            intersection1, intersection2, network
                        )
                        visual_connections[(i, j)] = connectivity
                    except Exception as conn_error:
                        logger.debug(f"視覚接続計算エラー ({i}, {j}): {conn_error}")
                        visual_connections[(i, j)] = 0.0

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
            logger.debug(f"視覚的接続性計算エラー: {e}")
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
                if visual_network.number_of_nodes() > 0:
                    largest_cc = max(nx.connected_components(visual_network), key=len)
                    subgraph = visual_network.subgraph(largest_cc)
                    if subgraph.number_of_nodes() > 1:
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    else:
                        avg_path_length = 0.0
                else:
                    avg_path_length = 0.0

            # 平均視覚的接続性
            if num_edges > 0:
                connection_scores = [data.get("visual_connectivity", 0)
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
            **self._get_empty_isovist_metrics()
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
