"""
グラフビルダー
パス: src/graph_builder.py

OSMネットワークをSpace Syntax解析用のグラフに変換
"""

import logging
import math
import time
from typing import List, Tuple, Dict, Any
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge, unary_union
import osmnx as ox

from .config_manager import GraphSettings


class GraphBuilder:
    """Space Syntax解析用グラフ構築クラス"""
    
    def __init__(self, settings: dict):
        """
        初期化
        
        Args:
            settings: グラフ設定辞書
        """
        self.logger = logging.getLogger(__name__)
        self.settings = GraphSettings(**settings)
        self.logger.info("GraphBuilder初期化完了")
    
    def build_axial_map(self, graph: nx.MultiDiGraph) -> nx.Graph:
        """
        Axial Map（軸線地図）の構築
        
        Args:
            graph: 元のOSMネットワークグラフ
            
        Returns:
            Axial Map グラフ
        """
        self.logger.info("Axial Map構築開始")
        start_time = time.time()
        
        try:
            # エッジをLineStringに変換
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
            
            # 道路の統合とAxial Line生成
            axial_lines = self._create_axial_lines(edges_gdf)
            
            # Axial Mapグラフ構築
            axial_graph = self._build_axial_graph(axial_lines)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Axial Map構築完了: {len(axial_graph.nodes)}軸線 "
                f"({elapsed_time:.2f}秒)"
            )
            
            return axial_graph
            
        except Exception as e:
            self.logger.error(f"Axial Map構築エラー: {e}")
            raise
    
    def build_segment_map(self, graph: nx.MultiDiGraph) -> nx.Graph:
        """
        Segment Map（セグメント地図）の構築
        
        Args:
            graph: 元のOSMネットワークグラフ
            
        Returns:
            Segment Map グラフ
        """
        self.logger.info("Segment Map構築開始")
        start_time = time.time()
        
        try:
            # セグメント分割
            segments = self._create_segments(graph)
            
            # Segment Mapグラフ構築
            segment_graph = self._build_segment_graph(segments)
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Segment Map構築完了: {len(segment_graph.nodes)}セグメント "
                f"({elapsed_time:.2f}秒)"
            )
            
            return segment_graph
            
        except Exception as e:
            self.logger.error(f"Segment Map構築エラー: {e}")
            raise
    
    def _create_axial_lines(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Axial Lineの生成
        
        Args:
            edges_gdf: エッジのGeoDataFrame
            
        Returns:
            Axial LineのGeoDataFrame
        """
        self.logger.debug("Axial Line生成開始")
        
        try:
            axial_lines = []
            processed_edges = set()
            
            for idx, edge in edges_gdf.iterrows():
                if idx in processed_edges:
                    continue
                
                # 同一道路名のエッジを収集
                road_name = edge.get('name', '')
                if pd.isna(road_name) or road_name == '':
                    road_name = f"unnamed_{idx}"
                
                # 同じ道路名のエッジを検索
                same_road_edges = edges_gdf[
                    (edges_gdf['name'] == road_name) | 
                    (edges_gdf['name'].isna() & pd.isna(road_name))
                ]
                
                if len(same_road_edges) == 1:
                    # 単一エッジの場合
                    axial_lines.append({
                        'geometry': edge['geometry'],
                        'name': road_name,
                        'length': edge['length'],
                        'original_edges': [idx]
                    })
                    processed_edges.add(idx)
                else:
                    # 複数エッジの統合
                    geometries = same_road_edges['geometry'].tolist()
                    merged_line = self._merge_lines(geometries)
                    
                    axial_lines.append({
                        'geometry': merged_line,
                        'name': road_name,
                        'length': merged_line.length,
                        'original_edges': same_road_edges.index.tolist()
                    })
                    
                    processed_edges.update(same_road_edges.index)
            
            # GeoDataFrame作成
            axial_gdf = gpd.GeoDataFrame(axial_lines, crs=edges_gdf.crs)
            
            # 最小長でフィルタリング
            min_length = self.settings.min_segment_length
            axial_gdf = axial_gdf[axial_gdf['length'] >= min_length]
            
            self.logger.debug(f"Axial Line生成完了: {len(axial_gdf)}本")
            return axial_gdf
            
        except Exception as e:
            self.logger.error(f"Axial Line生成エラー: {e}")
            raise
    
    def _merge_lines(self, geometries: List[LineString]) -> LineString:
        """
        LineStringの統合
        
        Args:
            geometries: LineStringのリスト
            
        Returns:
            統合されたLineString
        """
        try:
            if len(geometries) == 1:
                return geometries[0]
            
            # MultiLineStringに変換
            multi_line = MultiLineString(geometries)
            
            # 線の統合
            merged = linemerge(multi_line)
            
            if isinstance(merged, LineString):
                return merged
            elif isinstance(merged, MultiLineString):
                # 最長の線を選択
                return max(merged.geoms, key=lambda x: x.length)
            else:
                # フォールバック: 最初の線を返す
                return geometries[0]
                
        except Exception as e:
            self.logger.warning(f"線の統合エラー: {e}, 最初の線を使用")
            return geometries[0]
    
    def _build_axial_graph(self, axial_lines: gpd.GeoDataFrame) -> nx.Graph:
        """
        Axial Mapグラフの構築
        
        Args:
            axial_lines: Axial LineのGeoDataFrame
            
        Returns:
            Axial Mapグラフ
        """
        self.logger.debug("Axial Mapグラフ構築開始")
        
        try:
            graph = nx.Graph()
            
            # ノード追加
            for idx, line in axial_lines.iterrows():
                graph.add_node(idx, **{
                    'geometry': line['geometry'],
                    'name': line['name'],
                    'length': line['length'],
                    'original_edges': line['original_edges']
                })
            
            # エッジ追加（交差する軸線を接続）
            tolerance = self.settings.connection_tolerance
            
            for i, line1 in axial_lines.iterrows():
                for j, line2 in axial_lines.iterrows():
                    if i >= j:  # 重複回避
                        continue
                    
                    # 交差判定
                    if self._lines_intersect(
                        line1['geometry'], 
                        line2['geometry'], 
                        tolerance
                    ):
                        # 交差角度計算
                        angle = self._calculate_intersection_angle(
                            line1['geometry'], 
                            line2['geometry']
                        )
                        
                        graph.add_edge(i, j, angle=angle)
            
            self.logger.debug(f"Axial Mapグラフ構築完了: {len(graph.edges)}接続")
            return graph
            
        except Exception as e:
            self.logger.error(f"Axial Mapグラフ構築エラー: {e}")
            raise
    
    def _create_segments(self, graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        セグメントの作成
        
        Args:
            graph: 元のOSMネットワークグラフ
            
        Returns:
            セグメントのGeoDataFrame
        """
        self.logger.debug("セグメント作成開始")
        
        try:
            segments = []
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
            nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
            
            for (u, v, k), edge_data in graph.edges(keys=True, data=True):
                # エッジの幾何情報取得
                if 'geometry' in edge_data:
                    geometry = edge_data['geometry']
                else:
                    # ノード座標から線を作成
                    u_point = Point(nodes_gdf.loc[u, 'x'], nodes_gdf.loc[u, 'y'])
                    v_point = Point(nodes_gdf.loc[v, 'x'], nodes_gdf.loc[v, 'y'])
                    geometry = LineString([u_point, v_point])
                
                # 角度変化によるセグメント分割
                sub_segments = self._split_by_angle(geometry, u, v)
                
                for i, sub_geom in enumerate(sub_segments):
                    segments.append({
                        'geometry': sub_geom,
                        'length': sub_geom.length,
                        'original_edge': (u, v, k),
                        'segment_id': f"{u}_{v}_{k}_{i}",
                        'start_node': u if i == 0 else None,
                        'end_node': v if i == len(sub_segments)-1 else None,
                        **{key: val for key, val in edge_data.items() 
                           if key not in ['geometry']}
                    })
            
            # GeoDataFrame作成
            segments_gdf = gpd.GeoDataFrame(segments, crs=edges_gdf.crs)
            
            # 最小長でフィルタリング
            min_length = self.settings.min_segment_length
            segments_gdf = segments_gdf[segments_gdf['length'] >= min_length]
            
            self.logger.debug(f"セグメント作成完了: {len(segments_gdf)}個")
            return segments_gdf
            
        except Exception as e:
            self.logger.error(f"セグメント作成エラー: {e}")
            raise
    
    def _split_by_angle(self, geometry: LineString, start_node: int, end_node: int) -> List[LineString]:
        """
        角度変化によるLineStringの分割
        
        Args:
            geometry: 分割対象のLineString
            start_node: 開始ノード
            end_node: 終了ノード
            
        Returns:
            分割されたLineStringのリスト
        """
        try:
            if len(geometry.coords) <= 2:
                return [geometry]
            
            coords = list(geometry.coords)
            segments = []
            current_segment = [coords[0]]
            
            for i in range(1, len(coords) - 1):
                current_segment.append(coords[i])
                
                # 角度変化計算
                angle_change = self._calculate_angle_change(
                    coords[i-1], coords[i], coords[i+1]
                )
                
                # 閾値を超えた場合、新しいセグメント開始
                if angle_change > self.settings.angle_threshold:
                    if len(current_segment) >= 2:
                        segments.append(LineString(current_segment))
                    current_segment = [coords[i]]
            
            # 最後のセグメント
            current_segment.append(coords[-1])
            if len(current_segment) >= 2:
                segments.append(LineString(current_segment))
            
            return segments if segments else [geometry]
            
        except Exception:
            return [geometry]
    
    def _calculate_angle_change(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        """
        3点間の角度変化計算
        
        Args:
            p1, p2, p3: 座標点
            
        Returns:
            角度変化（度）
        """
        try:
            # ベクトル計算
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 角度計算
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # クランプ
            
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            
            return 180.0 - angle_deg  # 変化角度に変換
            
        except Exception:
            return 0.0
    
    def _build_segment_graph(self, segments: gpd.GeoDataFrame) -> nx.Graph:
        """
        Segment Mapグラフの構築
        
        Args:
            segments: セグメントのGeoDataFrame
            
        Returns:
            Segment Mapグラフ
        """
        self.logger.debug("Segment Mapグラフ構築開始")
        
        try:
            graph = nx.Graph()
            
            # ノード追加
            for idx, segment in segments.iterrows():
                graph.add_node(idx, **{
                    'geometry': segment['geometry'],
                    'length': segment['length'],
                    'segment_id': segment['segment_id'],
                    'original_edge': segment['original_edge']
                })
            
            # エッジ追加（接続するセグメントを結合）
            tolerance = self.settings.connection_tolerance
            
            for i, seg1 in segments.iterrows():
                for j, seg2 in segments.iterrows():
                    if i >= j:  # 重複回避
                        continue
                    
                    # 接続判定
                    if self._segments_connected(
                        seg1['geometry'], 
                        seg2['geometry'], 
                        tolerance
                    ):
                        # 角度差計算
                    angle_diff = self._calculate_segment_angle_difference(
                        seg1['geometry'], 
                        seg2['geometry']
                    )
                    
                    graph.add_edge(i, j, angle_difference=angle_diff)
            
            self.logger.debug(f"Segment Mapグラフ構築完了: {len(graph.edges)}接続")
            return graph
            
        except Exception as e:
            self.logger.error(f"Segment Mapグラフ構築エラー: {e}")
            raise
    
    def _lines_intersect(self, line1: LineString, line2: LineString, tolerance: float) -> bool:
        """
        2つの線が交差するかチェック
        
        Args:
            line1, line2: 対象線分
            tolerance: 許容距離
            
        Returns:
            交差の有無
        """
        try:
            # 交差判定
            if line1.intersects(line2):
                return True
            
            # 近接判定
            distance = line1.distance(line2)
            return distance <= tolerance
            
        except Exception:
            return False
    
    def _calculate_intersection_angle(self, line1: LineString, line2: LineString) -> float:
        """
        2つの線の交差角度計算
        
        Args:
            line1, line2: 対象線分
            
        Returns:
            交差角度（度）
        """
        try:
            # 線の方向ベクトル取得
            coords1 = list(line1.coords)
            coords2 = list(line2.coords)
            
            # 最初と最後の点から方向ベクトル計算
            v1 = (coords1[-1][0] - coords1[0][0], coords1[-1][1] - coords1[0][1])
            v2 = (coords2[-1][0] - coords2[0][0], coords2[-1][1] - coords2[0][1])
            
            # 角度計算
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 == 0 or mag2 == 0:
                return 90.0
            
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            angle_rad = math.acos(abs(cos_angle))
            angle_deg = math.degrees(angle_rad)
            
            # 0-90度の範囲に正規化
            return min(angle_deg, 180.0 - angle_deg)
            
        except Exception:
            return 90.0  # デフォルト値
    
    def _segments_connected(self, seg1: LineString, seg2: LineString, tolerance: float) -> bool:
        """
        2つのセグメントが接続されているかチェック
        
        Args:
            seg1, seg2: 対象セグメント
            tolerance: 許容距離
            
        Returns:
            接続の有無
        """
        try:
            # 端点の座標取得
            seg1_start = Point(seg1.coords[0])
            seg1_end = Point(seg1.coords[-1])
            seg2_start = Point(seg2.coords[0])
            seg2_end = Point(seg2.coords[-1])
            
            # 端点間の距離チェック
            connections = [
                seg1_start.distance(seg2_start),
                seg1_start.distance(seg2_end),
                seg1_end.distance(seg2_start),
                seg1_end.distance(seg2_end)
            ]
            
            return min(connections) <= tolerance
            
        except Exception:
            return False
    
    def _calculate_segment_angle_difference(self, seg1: LineString, seg2: LineString) -> float:
        """
        2つのセグメント間の角度差計算
        
        Args:
            seg1, seg2: 対象セグメント
            
        Returns:
            角度差（度）
        """
        try:
            # セグメントの方向ベクトル計算
            coords1 = list(seg1.coords)
            coords2 = list(seg2.coords)
            
            v1 = (coords1[-1][0] - coords1[0][0], coords1[-1][1] - coords1[0][1])
            v2 = (coords2[-1][0] - coords2[0][0], coords2[-1][1] - coords2[0][1])
            
            # 角度計算
            angle1 = math.atan2(v1[1], v1[0])
            angle2 = math.atan2(v2[1], v2[0])
            
            # 角度差計算
            diff = abs(angle1 - angle2)
            diff = math.degrees(diff)
            
            # 0-180度の範囲に正規化
            return min(diff, 360.0 - diff)
            
        except Exception:
            return 90.0  # デフォルト値
    
    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        グラフの統計情報取得
        
        Args:
            graph: 対象グラフ
            
        Returns:
            統計情報辞書
        """
        try:
            stats = {
                'node_count': len(graph.nodes),
                'edge_count': len(graph.edges),
                'is_connected': nx.is_connected(graph),
                'number_of_components': nx.number_connected_components(graph),
                'density': nx.density(graph),
            }
            
            if len(graph.nodes) > 0:
                # 次数統計
                degrees = [d for n, d in graph.degree()]
                stats.update({
                    'avg_degree': np.mean(degrees),
                    'max_degree': max(degrees),
                    'min_degree': min(degrees),
                })
                
                # 長さ統計（ノードに長さ情報がある場合）
                lengths = []
                for node, data in graph.nodes(data=True):
                    if 'length' in data:
                        lengths.append(data['length'])
                
                if lengths:
                    stats.update({
                        'total_length': sum(lengths),
                        'avg_length': np.mean(lengths),
                        'max_length': max(lengths),
                        'min_length': min(lengths)
                    })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"グラフ統計計算エラー: {e}")
            return {}
    
    def validate_graph(self, graph: nx.Graph) -> bool:
        """
        グラフの検証
        
        Args:
            graph: 検証対象グラフ
            
        Returns:
            検証結果
        """
        try:
            # 基本チェック
            if len(graph.nodes) == 0:
                self.logger.error("グラフにノードが存在しません")
                return False
            
            # 幾何情報チェック
            missing_geometry = 0
            for node, data in graph.nodes(data=True):
                if 'geometry' not in data:
                    missing_geometry += 1
            
            if missing_geometry > 0:
                self.logger.warning(f"{missing_geometry}個のノードに幾何情報がありません")
            
            # 孤立ノードチェック
            isolated_nodes = list(nx.isolates(graph))
            if isolated_nodes:
                self.logger.warning(f"{len(isolated_nodes)}個の孤立ノードがあります")
            
            # 連結性チェック
            if not nx.is_connected(graph):
                components = nx.number_connected_components(graph)
                self.logger.warning(f"グラフが{components}個の連結成分に分かれています")
            
            self.logger.info("グラフ検証完了")
            return True
            
        except Exception as e:
            self.logger.error(f"グラフ検証エラー: {e}")
            return False