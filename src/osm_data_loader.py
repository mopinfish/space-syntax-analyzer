"""
OpenStreetMapデータローダー
パス: src/osm_data_loader.py

OSMnxを使用してOpenStreetMapからネットワークデータを取得
"""

import logging
import time
from typing import Union, Tuple, Optional
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon

from .config_manager import OSMSettings


class OSMDataLoader:
    """OpenStreetMapデータ取得クラス"""
    
    def __init__(self, settings: dict):
        """
        初期化
        
        Args:
            settings: OSM設定辞書
        """
        self.logger = logging.getLogger(__name__)
        self.settings = OSMSettings(**settings)
        
        # OSMnx設定
        ox.settings.log_console = True
        ox.settings.use_cache = True
        ox.settings.timeout = self.settings.timeout
        
        if self.settings.memory:
            ox.settings.memory = self.settings.memory
        
        self.logger.info("OSMDataLoader初期化完了")
    
    def load_by_place(self, place: str) -> nx.MultiDiGraph:
        """
        地名によるネットワーク取得
        
        Args:
            place: 地名（例: "Shibuya, Tokyo, Japan"）
            
        Returns:
            ネットワークグラフ
        """
        self.logger.info(f"地名による取得開始: {place}")
        
        try:
            start_time = time.time()
            
            # ネットワーク取得
            graph = ox.graph_from_place(
                place,
                network_type=self.settings.network_type,
                simplify=self.settings.simplify,
                retain_all=self.settings.retain_all,
                truncate_by_edge=self.settings.truncate_by_edge
            )
            
            # グラフの前処理
            graph = self._preprocess_graph(graph)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"取得完了: {len(graph.nodes)}ノード, {len(graph.edges)}エッジ "
                f"({elapsed_time:.2f}秒)"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"地名による取得エラー: {e}")
            raise
    
    def load_by_bbox(self, bbox: Tuple[float, float, float, float]) -> nx.MultiDiGraph:
        """
        境界座標によるネットワーク取得
        
        Args:
            bbox: 境界座標 (south, west, north, east)
            
        Returns:
            ネットワークグラフ
        """
        south, west, north, east = bbox
        self.logger.info(f"境界座標による取得開始: {bbox}")
        
        try:
            start_time = time.time()
            
            # 境界チェック
            self._validate_bbox(bbox)
            
            # ネットワーク取得
            graph = ox.graph_from_bbox(
                north, south, east, west,
                network_type=self.settings.network_type,
                simplify=self.settings.simplify,
                retain_all=self.settings.retain_all,
                truncate_by_edge=self.settings.truncate_by_edge
            )
            
            # グラフの前処理
            graph = self._preprocess_graph(graph)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"取得完了: {len(graph.nodes)}ノード, {len(graph.edges)}エッジ "
                f"({elapsed_time:.2f}秒)"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"境界座標による取得エラー: {e}")
            raise
    
    def load_by_admin(self, admin_name: str) -> nx.MultiDiGraph:
        """
        行政区域によるネットワーク取得
        
        Args:
            admin_name: 行政区域名
            
        Returns:
            ネットワークグラフ
        """
        self.logger.info(f"行政区域による取得開始: {admin_name}")
        
        try:
            start_time = time.time()
            
            # 行政区域の境界取得
            gdf = ox.geocode_to_gdf(admin_name)
            
            if gdf.empty:
                raise ValueError(f"行政区域が見つかりません: {admin_name}")
            
            # ネットワーク取得
            graph = ox.graph_from_polygon(
                gdf.geometry.iloc[0],
                network_type=self.settings.network_type,
                simplify=self.settings.simplify,
                retain_all=self.settings.retain_all,
                truncate_by_edge=self.settings.truncate_by_edge
            )
            
            # グラフの前処理
            graph = self._preprocess_graph(graph)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"取得完了: {len(graph.nodes)}ノード, {len(graph.edges)}エッジ "
                f"({elapsed_time:.2f}秒)"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"行政区域による取得エラー: {e}")
            raise
    
    def load_by_polygon(self, polygon: Polygon) -> nx.MultiDiGraph:
        """
        ポリゴンによるネットワーク取得
        
        Args:
            polygon: 対象ポリゴン
            
        Returns:
            ネットワークグラフ
        """
        self.logger.info("ポリゴンによる取得開始")
        
        try:
            start_time = time.time()
            
            # ネットワーク取得
            graph = ox.graph_from_polygon(
                polygon,
                network_type=self.settings.network_type,
                simplify=self.settings.simplify,
                retain_all=self.settings.retain_all,
                truncate_by_edge=self.settings.truncate_by_edge
            )
            
            # グラフの前処理
            graph = self._preprocess_graph(graph)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"取得完了: {len(graph.nodes)}ノード, {len(graph.edges)}エッジ "
                f"({elapsed_time:.2f}秒)"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"ポリゴンによる取得エラー: {e}")
            raise
    
    def _preprocess_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        グラフの前処理
        
        Args:
            graph: 元のグラフ
            
        Returns:
            前処理済みグラフ
        """
        self.logger.debug("グラフ前処理開始")
        
        try:
            # 投影座標系への変換
            graph = ox.project_graph(graph)
            
            # エッジの長さ計算
            graph = ox.add_edge_lengths(graph)
            
            # ベアリング計算（方向性指標）
            graph = ox.add_edge_bearings(graph)
            
            # 接続性チェック
            if not nx.is_connected(graph.to_undirected()):
                self.logger.warning("グラフが非連結です。最大連結成分を取得します。")
                graph = ox.get_largest_component(graph, strongly=False)
            
            # ノード・エッジID正規化
            graph = ox.io._convert_node_attr_types(graph)
            graph = ox.io._convert_edge_attr_types(graph)
            
            self.logger.debug("グラフ前処理完了")
            return graph
            
        except Exception as e:
            self.logger.error(f"グラフ前処理エラー: {e}")
            raise
    
    def _validate_bbox(self, bbox: Tuple[float, float, float, float]):
        """
        境界座標の検証
        
        Args:
            bbox: 境界座標 (south, west, north, east)
        """
        south, west, north, east = bbox
        
        # 座標範囲チェック
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError(f"緯度が有効範囲外です: south={south}, north={north}")
        
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError(f"経度が有効範囲外です: west={west}, east={east}")
        
        # 順序チェック
        if south >= north:
            raise ValueError(f"南緯度 >= 北緯度: south={south}, north={north}")
        
        if west >= east:
            raise ValueError(f"西経度 >= 東経度: west={west}, east={east}")
        
        # 面積チェック（過大な領域の防止）
        area_deg = (north - south) * (east - west)
        if area_deg > self.settings.max_query_area_size / 111320**2:  # 概算
            raise ValueError(f"取得領域が大きすぎます: {area_deg:.6f}平方度")
    
    def get_network_info(self, graph: nx.MultiDiGraph) -> dict:
        """
        ネットワーク情報の取得
        
        Args:
            graph: ネットワークグラフ
            
        Returns:
            ネットワーク情報辞書
        """
        try:
            # 基本統計
            basic_stats = ox.basic_stats(graph)
            
            # 範囲情報
            nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
            bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            info = {
                'node_count': len(graph.nodes),
                'edge_count': len(graph.edges),
                'street_length_total': basic_stats.get('street_length_total', 0),
                'street_length_avg': basic_stats.get('street_length_avg', 0),
                'intersection_count': basic_stats.get('intersection_count', 0),
                'edge_length_total': basic_stats.get('edge_length_total', 0),
                'edge_length_avg': basic_stats.get('edge_length_avg', 0),
                'bounds': {
                    'west': bounds[0],
                    'south': bounds[1],
                    'east': bounds[2],
                    'north': bounds[3]
                },
                'crs': str(graph.graph.get('crs', 'unknown'))
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"ネットワーク情報取得エラー: {e}")
            return {}