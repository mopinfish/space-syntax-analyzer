"""
ネットワーク管理モジュール (OSMnx v2.0対応版)

このモジュールはOpenStreetMapからの道路ネットワークデータの取得と処理を担当します。
"""

import logging
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox

logger = logging.getLogger(__name__)


class NetworkManager:
    """ネットワーク取得・管理クラス (OSMnx v2.0対応)"""

    def __init__(self, network_type: str = "drive", width_threshold: float = 6.0):
        """
        ネットワーク管理クラスを初期化

        Args:
            network_type: ネットワークタイプ ("drive", "walk", "bike", "all", "all_public")
            width_threshold: 主要道路判定の幅閾値（メートル）
        """
        self.network_type = network_type
        self.width_threshold = width_threshold

        # OSMnx設定
        ox.settings.use_cache = True
        ox.settings.log_console = False

        logger.info(f"NetworkManager初期化: network_type={network_type}, width_threshold={width_threshold}")

    def get_network_from_bbox(self, bbox: tuple[float, float, float, float],
                             simplify: bool = False) -> nx.MultiDiGraph | None:
        """
        境界ボックスからネットワークを取得 (OSMnx v2.0対応)

        Args:
            bbox: (left, bottom, right, top) 形式の境界ボックス
            simplify: グラフを単純化するかどうか

        Returns:
            取得したNetworkXグラフ、失敗時はNone
        """
        try:
            logger.info(f"bbox からネットワーク取得開始: {bbox}")

            # OSMnx v2.0対応のAPI呼び出し
            G = ox.graph_from_bbox(
                bbox=bbox,
                network_type=self.network_type,
                simplify=simplify,
                retain_all=False,
                truncate_by_edge=False
            )

            logger.info(f"ネットワーク取得成功: {len(G.nodes)} ノード, {len(G.edges)} エッジ")
            return G

        except Exception as e:
            logger.error(f"bbox ネットワーク取得エラー: {e}")
            # 代替手段を試行
            return self._fallback_bbox_to_point(bbox, simplify)

    def get_network_from_place(self, place_name: str,
                              simplify: bool = False) -> nx.MultiDiGraph | None:
        """
        地名からネットワークを取得

        Args:
            place_name: 地名
            simplify: グラフを単純化するかどうか

        Returns:
            取得したNetworkXグラフ、失敗時はNone
        """
        try:
            logger.info(f"地名 '{place_name}' からネットワーク取得開始")

            G = ox.graph_from_place(
                place_name,
                network_type=self.network_type,
                simplify=simplify,
                retain_all=False
            )

            logger.info(f"地名からネットワーク取得成功: {len(G.nodes)} ノード, {len(G.edges)} エッジ")
            return G

        except Exception as e:
            logger.error(f"地名 '{place_name}' ネットワーク取得エラー: {e}")
            return None

    def get_network_from_point(self, center_point: tuple[float, float],
                              distance: float, simplify: bool = False) -> nx.MultiDiGraph | None:
        """
        中心点と距離からネットワークを取得

        Args:
            center_point: (lat, lon) 形式の中心座標
            distance: 距離（メートル）
            simplify: グラフを単純化するかどうか

        Returns:
            取得したNetworkXグラフ、失敗時はNone
        """
        try:
            logger.info(f"ポイント {center_point} (半径{distance}m) からネットワーク取得開始")

            G = ox.graph_from_point(
                center_point=center_point,
                dist=distance,
                dist_type="bbox",
                network_type=self.network_type,
                simplify=simplify
            )

            logger.info(f"ポイントからネットワーク取得成功: {len(G.nodes)} ノード, {len(G.edges)} エッジ")
            return G

        except Exception as e:
            logger.error(f"ポイント {center_point} ネットワーク取得エラー: {e}")
            return None

    def _fallback_bbox_to_point(self, bbox: tuple[float, float, float, float],
                               simplify: bool) -> nx.MultiDiGraph | None:
        """
        bbox取得失敗時の代替手段（中心点を使用）
        """
        try:
            # bboxの中心点を計算
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            # 距離を計算（緯度経度差から大まかに推定）
            lat_dist = abs(bbox[3] - bbox[1]) * 111000  # 緯度1度≈111km
            lon_dist = abs(bbox[2] - bbox[0]) * 111000 * np.cos(np.radians(center_lat))
            dist = max(lat_dist, lon_dist) / 2

            logger.info(f"代替手段でネットワーク取得: 中心({center_lat:.4f}, {center_lon:.4f}), 距離{dist:.0f}m")

            return self.get_network_from_point((center_lat, center_lon), dist, simplify)

        except Exception as e:
            logger.error(f"代替手段でもエラー: {e}")
            return None

    def safe_simplify_graph(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        安全にグラフを単純化（既に単純化されている場合はそのまま返す）

        Args:
            G: 単純化対象のグラフ

        Returns:
            単純化されたグラフ
        """
        try:
            # 既に単純化されているかチェック
            if hasattr(G.graph, 'simplified') and G.graph.get('simplified', False):
                logger.warning("グラフは既に単純化されています")
                return G

            # 単純化を実行
            logger.info("グラフを単純化中...")
            G_simplified = ox.simplify_graph(G)
            G_simplified.graph['simplified'] = True

            logger.info(f"グラフ単純化完了: {len(G.nodes)} -> {len(G_simplified.nodes)} ノード")
            return G_simplified

        except Exception as e:
            logger.error(f"グラフ単純化エラー: {e}")
            # 単純化に失敗した場合は元のグラフを返す
            return G

    def filter_major_roads(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        主要道路でフィルタリング

        Args:
            G: フィルタリング対象のグラフ

        Returns:
            主要道路のみのグラフ
        """
        try:
            logger.info("主要道路フィルタリング開始...")

            # 主要道路の判定基準
            major_highway_types = {
                'motorway', 'trunk', 'primary', 'secondary',
                'motorway_link', 'trunk_link', 'primary_link', 'secondary_link'
            }

            major_edges = []

            for u, v, key, data in G.edges(data=True, keys=True):
                is_major = False

                # 道路幅による判定
                width = data.get('width')
                if width:
                    try:
                        width_val = float(width) if isinstance(width, int | float | str) else 0
                        if width_val >= self.width_threshold:
                            is_major = True
                    except (ValueError, TypeError):
                        pass

                # 道路種別による判定
                highway = data.get('highway', '')
                if isinstance(highway, list):
                    highway = highway[0] if highway else ''

                if highway in major_highway_types:
                    is_major = True

                # レーン数による判定
                lanes = data.get('lanes')
                if lanes:
                    try:
                        lanes_val = int(lanes) if isinstance(lanes, int | str) else 0
                        if lanes_val >= 2:
                            is_major = True
                    except (ValueError, TypeError):
                        pass

                if is_major:
                    major_edges.append((u, v, key))

            # 主要道路のサブグラフを作成
            if major_edges:
                G_major = G.edge_subgraph(major_edges).copy()
                logger.info(f"主要道路フィルタリング完了: {len(major_edges)} / {len(G.edges)} エッジが主要道路")
            else:
                logger.warning("主要道路が見つかりません。元のグラフを返します")
                G_major = G

            return G_major

        except Exception as e:
            logger.error(f"主要道路フィルタリングエラー: {e}")
            return G

    def export_network(self, G: nx.MultiDiGraph, filepath: str,
                      format_type: str = "graphml") -> bool:
        """
        ネットワークをファイルに出力

        Args:
            G: 出力するグラフ
            filepath: 出力先パス
            format_type: ファイル形式 ("graphml", "geojson", "shapefile")

        Returns:
            出力成功時True、失敗時False
        """
        try:
            logger.info(f"ネットワーク出力開始: {filepath} ({format_type})")

            if format_type.lower() == "graphml":
                ox.save_graphml(G, filepath)

            elif format_type.lower() == "geojson":
                gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
                gdf_edges.to_file(filepath, driver="GeoJSON")

            elif format_type.lower() in ["shapefile", "shp"]:
                gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
                gdf_edges.to_file(filepath)

            else:
                raise ValueError(f"サポートされていないフォーマット: {format_type}")

            logger.info(f"ネットワーク出力完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"ネットワーク出力エラー: {e}")
            return False

    def calculate_network_stats(self, G: nx.MultiDiGraph) -> dict[str, Any]:
        """
        ネットワークの基本統計を計算

        Args:
            G: 統計を計算するグラフ

        Returns:
            統計情報の辞書
        """
        try:
            stats = {}

            # 基本情報
            stats['node_count'] = len(G.nodes)
            stats['edge_count'] = len(G.edges)

            if len(G.nodes) == 0:
                return stats

            # 次数統計
            degrees = dict(G.degree())
            stats['avg_degree'] = np.mean(list(degrees.values()))
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())

            # 連結性
            if nx.is_connected(G.to_undirected()):
                stats['is_connected'] = True
                stats['connectivity_ratio'] = 1.0
            else:
                # 最大連結成分
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                stats['is_connected'] = False
                stats['largest_component_size'] = len(largest_cc)
                stats['connectivity_ratio'] = len(largest_cc) / len(G.nodes)

            # 密度
            stats['density'] = nx.density(G)

            logger.info(f"ネットワーク統計計算完了: {stats['node_count']} ノード")
            return stats

        except Exception as e:
            logger.error(f"ネットワーク統計計算エラー: {e}")
            return {'error': str(e)}


def create_bbox_from_center(lat: float, lon: float,
                           distance_km: float = 1.0) -> tuple[float, float, float, float]:
    """
    中心点から指定距離のbboxを作成 (OSMnx v2.0対応)

    Args:
        lat: 緯度
        lon: 経度
        distance_km: 距離（km）

    Returns:
        (left, bottom, right, top) 形式のbbox
    """
    try:
        # OSMnx v2.0のutils_geo.bbox_from_point を使用
        bbox = ox.utils_geo.bbox_from_point(
            point=(lat, lon),
            dist=distance_km * 1000  # kmをmに変換
        )
        return bbox

    except Exception as e:
        logger.error(f"bbox作成エラー: {e}")
        # フォールバック: 手動計算
        d = distance_km / 111.0  # 緯度経度への近似変換
        return (lon - d, lat - d, lon + d, lat + d)


def get_network_from_query(query: str | tuple[float, float, float, float],
                          network_type: str = "drive",
                          simplify: bool = False) -> nx.MultiDiGraph | None:
    """
    クエリ（地名またはbbox）からネットワークを取得する便利関数

    Args:
        query: 地名または(left, bottom, right, top)のbbox
        network_type: ネットワークタイプ
        simplify: 単純化するかどうか

    Returns:
        取得したNetworkXグラフ、失敗時はNone
    """
    manager = NetworkManager(network_type)

    if isinstance(query, str):
        return manager.get_network_from_place(query, simplify)
    elif isinstance(query, tuple | list) and len(query) == 4:
        return manager.get_network_from_bbox(tuple(query), simplify)
    else:
        logger.error(f"不正なクエリ形式: {query}")
        return None
