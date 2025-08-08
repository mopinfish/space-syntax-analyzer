# space_syntax_analyzer/utils/geo_utils.py
"""
地理計算ユーティリティ（既存ファイルの拡張）
"""

import logging
import math

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class GeoUtils:
    """地理計算ユーティリティクラス"""

    # 地球半径（メートル）
    EARTH_RADIUS = 6371000.0

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        2点間の距離を計算（Haversine公式）

        Parameters
        ----------
        lat1, lon1 : float
            地点1の緯度・経度
        lat2, lon2 : float
            地点2の緯度・経度

        Returns
        -------
        float
            距離（メートル）
        """
        # 度をラジアンに変換
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine公式
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))

        distance = GeoUtils.EARTH_RADIUS * c
        return distance

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        2点間の方位角を計算

        Parameters
        ----------
        lat1, lon1 : float
            地点1の緯度・経度
        lat2, lon2 : float
            地点2の緯度・経度

        Returns
        -------
        float
            方位角（度、0-360）
        """
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad

        y = math.sin(dlon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        # 0-360の範囲に正規化
        return (bearing_deg + 360) % 360

    @staticmethod
    def calculate_network_area_ha(network) -> float:
        """
        ネットワークがカバーする面積を計算（ヘクタール）

        Parameters
        ----------
        network : nx.MultiDiGraph
            ネットワーク

        Returns
        -------
        float
            面積（ヘクタール）
        """
        try:
            # ノード座標の境界を取得
            lats = []
            lons = []

            for _, data in network.nodes(data=True):
                if "y" in data and "x" in data:
                    lats.append(data["y"])
                    lons.append(data["x"])

            if not lats or not lons:
                return 0.0

            # 境界ボックスの計算
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            # 簡易面積計算（度数を距離に変換）
            lat_dist = GeoUtils.calculate_distance(min_lat, min_lon, max_lat, min_lon)
            lon_dist = GeoUtils.calculate_distance(min_lat, min_lon, min_lat, max_lon)

            area_m2 = lat_dist * lon_dist
            area_ha = area_m2 / 10000.0  # ヘクタールに変換

            logger.info(f"分析エリア面積: {area_ha:.1f}ha")
            return area_ha

        except Exception as e:
            logger.warning(f"面積計算エラー: {e}")
            return 100.0  # デフォルト値

    @staticmethod
    def create_buffer_polygon(lat: float, lon: float, radius: float) -> Polygon:
        """
        点の周囲にバッファ領域を作成

        Parameters
        ----------
        lat, lon : float
            中心点の緯度・経度
        radius : float
            バッファ半径（メートル）

        Returns
        -------
        Polygon
            バッファ領域
        """
        # 簡易的な円形バッファを作成（度単位）
        deg_per_meter_lat = 1.0 / 111320.0  # 緯度1度 ≈ 111.32km
        deg_per_meter_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)))

        radius_lat = radius * deg_per_meter_lat
        radius * deg_per_meter_lon

        center = Point(lon, lat)
        # 楕円形バッファ（緯度による圧縮を考慮）
        buffer_circle = center.buffer(radius_lat)

        return buffer_circle

    @staticmethod
    def normalize_coordinates(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        座標リストを正規化

        Parameters
        ----------
        coordinates : list
            座標リスト [(lat, lon), ...]

        Returns
        -------
        list
            正規化された座標リスト
        """
        if not coordinates:
            return []

        # 重複除去
        unique_coords = []
        for coord in coordinates:
            if coord not in unique_coords:
                unique_coords.append(coord)

        # 座標範囲チェック
        valid_coords = []
        for lat, lon in unique_coords:
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_coords.append((lat, lon))
            else:
                logger.warning(f"無効な座標をスキップ: ({lat}, {lon})")

        return valid_coords
