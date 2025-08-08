"""
ネットワーク管理モジュール - NetworkManager

OSMnxを利用した道路ネットワークの取得と前処理を行います。
"""

from __future__ import annotations

import logging

import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class NetworkManager:
    """道路ネットワークの取得と管理を行うクラス"""

    def __init__(
        self,
        width_threshold: float = 4.0,
        network_type: str = "drive",
        crs: str = "EPSG:4326",
    ) -> None:
        """
        NetworkManagerを初期化

        Args:
            width_threshold: 道路幅員の閾値（メートル）
            network_type: OSMnxのネットワークタイプ
            crs: 座標参照系
        """
        self.width_threshold = width_threshold
        self.network_type = network_type
        self.crs = crs

        # OSMnxの設定
        ox.settings.use_cache = True
        ox.settings.log_console = False

    def get_network(
        self,
        location: str | tuple[float, float, float, float] | Polygon,
        network_filter: str = "both",
    ) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        """
        指定された場所の道路ネットワークを取得

        Args:
            location: 場所の指定
            network_filter: ネットワークフィルター（'major', 'all', 'both'）

        Returns:
            Tuple[主要道路ネットワーク, 全道路ネットワーク]
        """
        try:
            # 場所の種類に応じてネットワークを取得
            if isinstance(location, str):
                graph = ox.graph_from_place(location, network_type=self.network_type)
            elif isinstance(location, tuple) and len(location) == 4:
                # bbox: (north, south, east, west)
                graph = ox.graph_from_bbox(
                    location[0],
                    location[1],
                    location[2],
                    location[3],
                    network_type=self.network_type,
                )
            elif isinstance(location, Polygon):
                graph = ox.graph_from_polygon(location, network_type=self.network_type)
            else:
                raise ValueError("Invalid location format")

            # 座標系を統一
            graph = ox.project_graph(graph, to_crs=self.crs)

            # ネットワークの前処理
            graph = self._preprocess_network(graph)

            # 道路幅員による分類
            if network_filter == "major":
                major_network = self._filter_major_roads(graph)
                return major_network, major_network
            elif network_filter == "all":
                return graph, graph
            else:  # both
                major_network = self._filter_major_roads(graph)
                return major_network, graph

        except Exception as e:
            logger.error(f"ネットワーク取得エラー: {e}")
            raise

    def _preprocess_network(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        ネットワークの前処理

        Args:
            graph: 原始ネットワークグラフ

        Returns:
            前処理済みネットワークグラフ
        """
        # 自己ループとマルチエッジの除去
        graph = ox.simplify_graph(graph)

        # 未接続コンポーネントの処理（最大のコンポーネントのみ保持）
        if not nx.is_strongly_connected(graph):
            largest_cc = max(nx.strongly_connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

        # 無向グラフに変換（スペースシンタックス分析用）
        graph = nx.to_undirected(graph)

        return graph

    def _filter_major_roads(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        主要道路（幅員4m以上）のみを抽出

        Args:
            graph: 全道路ネットワーク

        Returns:
            主要道路ネットワーク
        """
        major_edges = []

        for u, v, data in graph.edges(data=True):
            # 道路幅員の判定
            width = self._estimate_road_width(data)
            if width >= self.width_threshold:
                major_edges.append((u, v, data))

        # 主要道路のみのグラフを作成
        major_graph = nx.MultiGraph()
        major_graph.add_nodes_from(graph.nodes(data=True))
        major_graph.add_edges_from(major_edges)

        # 孤立ノードの除去
        isolated_nodes = list(nx.isolates(major_graph))
        major_graph.remove_nodes_from(isolated_nodes)

        return major_graph

    def _estimate_road_width(self, edge_data: dict) -> float:
        """
        エッジデータから道路幅員を推定

        Args:
            edge_data: エッジの属性データ

        Returns:
            推定道路幅員（メートル）
        """
        # 幅員データが直接ある場合
        if 'width' in edge_data and edge_data['width'] is not None:
            try:
                width_str = str(edge_data['width'])
                # 数値部分を抽出
                import re

                width_match = re.search(r'(\d+\.?\d*)', width_str)
                if width_match:
                    return float(width_match.group(1))
            except (ValueError, AttributeError):
                pass

        # est_widthがある場合
        if 'est_width' in edge_data and edge_data['est_width'] is not None:
            try:
                return float(edge_data['est_width'])
            except (ValueError, TypeError):
                pass

        # 車線数から推定
        if 'lanes' in edge_data and edge_data['lanes'] is not None:
            try:
                lanes = int(edge_data['lanes'])
                return lanes * 3.0  # 1車線約3mと仮定
            except (ValueError, TypeError):
                pass

        # highway タグから推定
        highway_widths = {
            'motorway': 12.0,
            'trunk': 10.0,
            'primary': 8.0,
            'secondary': 6.0,
            'tertiary': 5.0,
            'residential': 4.0,
            'service': 3.0,
            'footway': 1.5,
            'path': 1.0,
        }

        highway_type = edge_data.get('highway', 'residential')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        return highway_widths.get(highway_type, 4.0)

    def calculate_area_ha(self, graph: nx.Graph) -> float:
        """
        ネットワークの外接矩形から面積を計算（ヘクタール）

        Args:
            graph: ネットワークグラフ

        Returns:
            面積（ヘクタール）
        """
        if len(graph.nodes) == 0:
            return 0.0

        # ノードの座標を取得
        nodes_gdf = ox.graph_to_gdfs(graph, edges=False)

        # 外接矩形を計算
        bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]

        # 面積計算（平方メートル → ヘクタール）
        width_m = bounds[2] - bounds[0]
        height_m = bounds[3] - bounds[1]
        area_m2 = width_m * height_m
        area_ha = area_m2 / 10000  # 1ヘクタール = 10,000平方メートル

        return area_ha

    def export_network(
        self, graph: nx.Graph, output_path: str, format_type: str = "geojson"
    ) -> None:
        """
        ネットワークをファイルに出力

        Args:
            graph: ネットワークグラフ
            output_path: 出力パス
            format_type: 出力形式（'geojson', 'shapefile', 'graphml'）
        """
        try:
            if format_type.lower() == "geojson":
                # GeoDataFrameに変換してGeoJSON出力
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
                edges_gdf.to_file(output_path, driver="GeoJSON", encoding="utf-8")

            elif format_type.lower() == "shapefile":
                # Shapefile出力
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
                edges_gdf.to_file(output_path, encoding="utf-8")

            elif format_type.lower() == "graphml":
                # GraphML出力
                ox.save_graphml(graph, output_path)

            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"ネットワーク出力エラー: {e}")
            raise
