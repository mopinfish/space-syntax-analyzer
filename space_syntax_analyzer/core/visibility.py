"""
最適化された可視領域分析モジュール - VisibilityAnalyzer

パフォーマンスを大幅に改善し、視覚的接続性分析の処理時間を短縮
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class VisibilityAnalyzer:
    """最適化された可視領域分析を行うクラス"""

    def __init__(self, visibility_radius: float = 100.0, max_intersections: int = 50):
        """
        VisibilityAnalyzerを初期化

        Args:
            visibility_radius: 可視領域半径（メートル）
            max_intersections: 処理する最大交差点数（パフォーマンス制限）
        """
        self.visibility_radius = visibility_radius
        self.max_intersections = max_intersections
        self.obstacles_cache = None  # 障害物キャッシュ

    def analyze_isovist(self, observation_point: tuple[float, float],
                       network: nx.Graph, radius: float | None = None) -> dict[str, Any]:
        """
        単一点のIsovist分析を実行

        Args:
            observation_point: 観測点の座標
            network: 道路ネットワーク
            radius: 可視領域半径

        Returns:
            Isovist分析結果
        """
        if radius is None:
            radius = self.visibility_radius

        try:
            logger.debug(f"Isovist分析開始: {observation_point}")

            # 観測点の妥当性チェック
            if not self._is_valid_point(observation_point):
                logger.warning(f"無効な観測点: {observation_point}")
                return self._empty_isovist_result(observation_point)

            # 可視領域ポリゴンの計算（簡略化）
            visibility_polygon = self._calculate_simplified_visibility_polygon(
                observation_point, network, radius
            )

            # Isovistメトリクスの計算
            metrics = self._calculate_isovist_metrics(visibility_polygon)

            return {
                "observation_point": observation_point,
                "visibility_polygon": visibility_polygon,
                "obstacles": [],  # 簡略化のため空リスト
                **metrics
            }

        except Exception as e:
            logger.warning(f"Isovist分析エラー ({observation_point}): {e}")
            return self._empty_isovist_result(observation_point)

    def analyze_visibility_field(self, network: nx.Graph,
                                sampling_distance: float = 25.0,
                                max_points: int = 100) -> dict[str, Any]:
        """
        最適化された可視領域フィールド分析

        Args:
            network: 道路ネットワーク
            sampling_distance: サンプリング間隔
            max_points: 最大サンプリング点数

        Returns:
            可視領域フィールド分析結果
        """
        try:
            logger.info("可視領域フィールド分析開始")
            start_time = time.time()

            # サンプリング点の生成（制限付き）
            sampling_points = self._generate_limited_sampling_points(
                network, sampling_distance, max_points
            )

            if not sampling_points:
                logger.warning("サンプリング点が生成できませんでした")
                return self._empty_visibility_field_result()

            logger.info(f"サンプリング点数: {len(sampling_points)}")

            # 並列処理でIsovist分析を実行
            isovist_results = self._parallel_isovist_analysis(sampling_points, network)

            # フィールド統計の計算
            field_statistics = self._calculate_field_statistics(isovist_results)

            # 変動性指標の計算
            variability_metrics = self._calculate_variability_metrics(isovist_results)

            end_time = time.time()
            logger.info(f"可視領域フィールド分析完了: {len(sampling_points)}点 "
                       f"(実行時間: {end_time - start_time:.1f}秒)")

            return {
                "sampling_points": sampling_points,
                "isovist_results": isovist_results,
                "field_statistics": field_statistics,
                "variability_metrics": variability_metrics,
            }

        except Exception as e:
            logger.error(f"可視領域フィールド分析エラー: {e}")
            return self._empty_visibility_field_result()

    def analyze_visual_connectivity(self, network: nx.Graph) -> dict[str, Any]:
        """
        最適化された視覚的接続性分析

        Args:
            network: 道路ネットワーク

        Returns:
            視覚的接続性分析結果
        """
        try:
            logger.info("視覚的接続性分析開始")
            start_time = time.time()

            # 主要交差点の抽出（制限付き）
            major_intersections = self._extract_limited_major_intersections(network)

            if not major_intersections:
                logger.warning("主要交差点が見つかりませんでした")
                return self._empty_visual_connectivity_result()

            logger.info(f"分析対象交差点数: {len(major_intersections)}")

            # 効率的な視覚的接続性計算
            visual_connections = self._calculate_efficient_visual_connections(
                major_intersections, network
            )

            # 視覚的接続性ネットワークの構築
            visual_network = self._build_visual_network(
                major_intersections, visual_connections
            )

            # ネットワーク指標の計算
            network_metrics = self._calculate_visual_network_metrics(visual_network)

            end_time = time.time()
            logger.info(f"視覚的接続性分析完了: {len(major_intersections)}交差点 "
                       f"(実行時間: {end_time - start_time:.1f}秒)")

            return {
                "major_intersections": major_intersections,
                "visual_connections": visual_connections,
                "visual_network": visual_network,
                "network_metrics": network_metrics,
            }

        except Exception as e:
            logger.error(f"視覚的接続性分析エラー: {e}")
            return self._empty_visual_connectivity_result()

    def _generate_limited_sampling_points(self, network: nx.Graph,
                                        sampling_distance: float,
                                        max_points: int) -> list[tuple[float, float]]:
        """
        制限付きサンプリング点生成

        Args:
            network: 道路ネットワーク
            sampling_distance: サンプリング間隔
            max_points: 最大点数

        Returns:
            サンプリング点リスト
        """
        try:
            # ネットワークの境界を取得
            bounds = self._get_network_bounds(network)
            if not bounds:
                return []

            min_x, min_y, max_x, max_y = bounds

            # グリッド点を生成
            x_points = np.arange(min_x, max_x, sampling_distance)
            y_points = np.arange(min_y, max_y, sampling_distance)

            sampling_points = []
            for x in x_points:
                for y in y_points:
                    if len(sampling_points) >= max_points:
                        break
                    sampling_points.append((float(x), float(y)))
                if len(sampling_points) >= max_points:
                    break

            return sampling_points

        except Exception as e:
            logger.warning(f"サンプリング点生成エラー: {e}")
            return []

    def _extract_limited_major_intersections(self, network: nx.Graph) -> list[tuple[float, float]]:
        """
        制限付き主要交差点抽出

        Args:
            network: 道路ネットワーク

        Returns:
            主要交差点の座標リスト（最大制限付き）
        """
        intersections = []

        # 次数でソートして重要な交差点を優先
        nodes_by_degree = sorted(
            network.nodes(data=True),
            key=lambda x: network.degree[x[0]],
            reverse=True
        )

        for node, data in nodes_by_degree:
            if len(intersections) >= self.max_intersections:
                break

            # 次数3以上を交差点とする
            if network.degree[node] >= 3 and "x" in data and "y" in data:
                intersections.append((data["x"], data["y"]))

        return intersections

    def _calculate_efficient_visual_connections(self, intersections: list[tuple[float, float]],
                                              network: nx.Graph) -> dict[tuple[int, int], float]:
        """
        効率的な視覚的接続性計算

        Args:
            intersections: 交差点リスト
            network: 道路ネットワーク

        Returns:
            視覚的接続性スコア
        """
        visual_connections = {}

        # 距離制限で計算対象を削減
        max_sight_distance = self.visibility_radius * 2

        for i, intersection1 in enumerate(intersections):
            for j, intersection2 in enumerate(intersections[i+1:], i+1):
                # 距離チェック
                dist = distance.euclidean(intersection1, intersection2)
                if dist > max_sight_distance:
                    visual_connections[(i, j)] = 0.0
                    continue

                try:
                    # 簡略化された視覚的接続性計算
                    connectivity = self._calculate_simplified_visual_connection(
                        intersection1, intersection2, dist
                    )
                    visual_connections[(i, j)] = connectivity
                except Exception as conn_error:
                    logger.debug(f"視覚接続計算エラー ({i}, {j}): {conn_error}")
                    visual_connections[(i, j)] = 0.0

        return visual_connections

    def _calculate_simplified_visual_connection(self, point1: tuple[float, float],
                                              point2: tuple[float, float],
                                              distance_between: float) -> float:
        """
        簡略化された視覚的接続性計算

        Args:
            point1, point2: 分析対象の2点
            distance_between: 2点間の距離

        Returns:
            視覚的接続性スコア（0-1）
        """
        try:
            # 距離ベースの簡略化計算
            max_visible_distance = self.visibility_radius

            if distance_between > max_visible_distance:
                return 0.0

            # 距離減衰モデル
            distance_factor = 1.0 - (distance_between / max_visible_distance)

            # 基本的な視覚的接続性スコア
            base_connectivity = 0.8  # 障害物がない場合の基本値

            connectivity_score = base_connectivity * distance_factor

            return max(0.0, min(1.0, connectivity_score))

        except Exception as e:
            logger.debug(f"簡略化視覚的接続性計算エラー: {e}")
            return 0.0

    def _parallel_isovist_analysis(self, sampling_points: list[tuple[float, float]],
                                 network: nx.Graph) -> list[dict[str, Any]]:
        """
        並列処理によるIsovist分析

        Args:
            sampling_points: サンプリング点リスト
            network: 道路ネットワーク

        Returns:
            Isovist分析結果リスト
        """
        isovist_results = []

        try:
            # 小さなデータセットでは逐次処理
            if len(sampling_points) <= 20:
                for point in sampling_points:
                    result = self.analyze_isovist(point, network)
                    isovist_results.append(result)
            else:
                # 並列処理
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_point = {
                        executor.submit(self.analyze_isovist, point, network): point
                        for point in sampling_points
                    }

                    for future in as_completed(future_to_point):
                        try:
                            result = future.result(timeout=30)  # 30秒タイムアウト
                            isovist_results.append(result)
                        except Exception as e:
                            point = future_to_point[future]
                            logger.warning(f"並列Isovist分析エラー ({point}): {e}")
                            isovist_results.append(self._empty_isovist_result(point))

        except Exception as e:
            logger.warning(f"並列Isovist分析エラー: {e}")
            # フォールバック: 逐次処理
            for point in sampling_points:
                try:
                    result = self.analyze_isovist(point, network)
                    isovist_results.append(result)
                except Exception as point_error:
                    logger.debug(f"逐次Isovist分析エラー ({point}): {point_error}")
                    isovist_results.append(self._empty_isovist_result(point))

        return isovist_results

    def _calculate_simplified_visibility_polygon(self, observation_point: tuple[float, float],
                                               network: nx.Graph,
                                               radius: float) -> Polygon:
        """
        簡略化された可視領域ポリゴン計算

        Args:
            observation_point: 観測点
            network: 道路ネットワーク
            radius: 可視半径

        Returns:
            可視領域ポリゴン
        """
        try:
            # 簡略化: 円形の可視領域として近似
            center = Point(observation_point)
            visibility_circle = center.buffer(radius)

            # より高精度が必要な場合は、ここで障害物との交差計算を追加
            return visibility_circle

        except Exception as e:
            logger.debug(f"簡略化可視領域計算エラー: {e}")
            return Point(observation_point).buffer(0)

    def _calculate_isovist_metrics(self, visibility_polygon: Polygon) -> dict[str, float]:
        """
        Isovistメトリクスの計算

        Args:
            visibility_polygon: 可視領域ポリゴン

        Returns:
            Isovistメトリクス
        """
        try:
            area = visibility_polygon.area
            perimeter = visibility_polygon.length

            # 境界の複雑さ
            compactness = 4 * np.pi * area / perimeter ** 2 if area > 0 else 0.0

            # アスペクト比の近似
            bounds = visibility_polygon.bounds
            if len(bounds) == 4:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = max(width, height) / max(min(width, height), 0.001)
            else:
                aspect_ratio = 1.0

            return {
                "visible_area": area,
                "perimeter": perimeter,
                "compactness": compactness,
                "aspect_ratio": aspect_ratio,
                "drift_angle": 0.0,  # 簡略化
                "drift_magnitude": 0.0,  # 簡略化
            }

        except Exception as e:
            logger.debug(f"Isovistメトリクス計算エラー: {e}")
            return self._get_empty_isovist_metrics()

    def _calculate_field_statistics(self, isovist_results: list[dict[str, Any]]) -> dict[str, float]:
        """
        フィールド統計の計算

        Args:
            isovist_results: Isovist分析結果リスト

        Returns:
            フィールド統計
        """
        try:
            if not isovist_results:
                return {}

            # 可視面積の統計
            visible_areas = [result.get("visible_area", 0) for result in isovist_results]

            return {
                "total_sampling_points": len(isovist_results),
                "mean_visible_area": np.mean(visible_areas),
                "std_visible_area": np.std(visible_areas),
                "min_visible_area": np.min(visible_areas),
                "max_visible_area": np.max(visible_areas),
                "mean_compactness": np.mean([result.get("compactness", 0) for result in isovist_results]),
            }

        except Exception as e:
            logger.warning(f"フィールド統計計算エラー: {e}")
            return {}

    def _calculate_variability_metrics(self, isovist_results: list[dict[str, Any]]) -> dict[str, float]:
        """
        変動性指標の計算

        Args:
            isovist_results: Isovist分析結果リスト

        Returns:
            変動性指標
        """
        try:
            if not isovist_results:
                return {}

            visible_areas = [result.get("visible_area", 0) for result in isovist_results]

            if not visible_areas:
                return {}

            # 変動係数
            mean_area = np.mean(visible_areas)
            std_area = np.std(visible_areas)

            coefficient_of_variation = std_area / mean_area if mean_area > 0 else 0.0

            # エントロピー（簡略化）
            entropy = self._calculate_diversity_index(visible_areas)

            return {
                "coefficient_of_variation": coefficient_of_variation,
                "entropy": entropy,
                "range": np.max(visible_areas) - np.min(visible_areas) if visible_areas else 0,
            }

        except Exception as e:
            logger.warning(f"変動性指標計算エラー: {e}")
            return {}

    def _calculate_diversity_index(self, values: list[float]) -> float:
        """
        多様性指標（エントロピー）の計算

        Args:
            values: 値のリスト

        Returns:
            エントロピー値
        """
        try:
            if not values:
                return 0.0

            # ヒストグラムビンの計算
            bins, _ = np.histogram(values, bins=10)
            total = np.sum(bins)

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
        connection_threshold = 0.3  # 閾値を下げて接続を増やす

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
            if nx.is_connected(visual_network) and num_nodes > 1:
                avg_path_length = nx.average_shortest_path_length(visual_network)
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

    # ユーティリティメソッド
    def _is_valid_point(self, point: tuple[float, float]) -> bool:
        """点の妥当性をチェック"""
        try:
            x, y = point
            return isinstance(x, int | float) and isinstance(y, int | float)
        except (ValueError, TypeError):
            return False

    def _get_network_bounds(self, network: nx.Graph) -> tuple[float, float, float, float] | None:
        """ネットワークの境界を取得"""
        try:
            xs = [data.get("x") for _, data in network.nodes(data=True) if "x" in data]
            ys = [data.get("y") for _, data in network.nodes(data=True) if "y" in data]

            if not xs or not ys:
                return None

            return (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            return None

    def _get_empty_isovist_metrics(self) -> dict[str, float]:
        """空のIsovistメトリクスを返す"""
        return {
            "visible_area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "aspect_ratio": 1.0,
            "drift_angle": 0.0,
            "drift_magnitude": 0.0,
        }

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
