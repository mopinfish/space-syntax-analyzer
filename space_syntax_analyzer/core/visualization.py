# space_syntax_analyzer/core/visualization.py (完全修正版)

"""
ネットワーク可視化モジュール

Space Syntax分析結果の効果的な可視化機能を提供
"""

# 標準インポート
import logging
from typing import Any

# matplotlib設定とインポート
import matplotlib

# サードパーティインポート
import networkx as nx
import pandas as pd

matplotlib.use("Agg")  # GUIなしバックエンドを設定
import matplotlib.pyplot as plt

# 背景地図用インポート
try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False

# 地理空間データ処理用インポート
try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# 日本語フォント設定
try:
    import japanize_matplotlib  # noqa: F401

    # japanize_matplotlibは副作用（日本語フォント設定）のためのインポート
    JAPANESE_FONT_AVAILABLE = True
    print("✅ 日本語フォント設定完了")
except ImportError:
    # フォールバック：英語フォント
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    JAPANESE_FONT_AVAILABLE = False
    print("⚠️ 日本語フォントが利用できません。英語フォントを使用します。")

plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """ネットワーク可視化クラス"""

    def __init__(self):
        """可視化器を初期化"""
        self.japanese_available = JAPANESE_FONT_AVAILABLE
        self.contextily_available = CONTEXTILY_AVAILABLE
        self.geopandas_available = GEOPANDAS_AVAILABLE

        if not self.contextily_available:
            logger.warning("contextily が利用できません。背景地図なしで描画します。")
        if not self.geopandas_available:
            logger.warning("geopandas が利用できません。地理座標変換機能が制限されます。")

        logger.info("NetworkVisualizer初期化完了")

    def _add_basemap(self, ax: plt.Axes, network: nx.MultiDiGraph,
                    alpha: float = 0.6, zoom_level: int = 15) -> bool:
        """
        OpenStreetMapの背景地図を追加

        Args:
            ax: matplotlib軸
            network: ネットワーク（座標範囲の取得用）
            alpha: 背景地図の透過率 (0-1)
            zoom_level: ズームレベル

        Returns:
            背景地図追加成功時True
        """
        try:
            if not self.contextily_available or not self.geopandas_available:
                return False

            import numpy as np
            from pyproj import Transformer

            # ネットワークの座標範囲を取得
            coords = []
            for _node, data in network.nodes(data=True):
                if "x" in data and "y" in data:
                    coords.append([data["x"], data["y"]])

            if not coords:
                return False

            coords = np.array(coords)

            # 座標系の判定と変換
            # OSMnxは通常UTM座標系またはWeb Mercator (EPSG:3857)で座標を提供
            # まず座標の範囲から座標系を推定
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()

            # Web Mercator (EPSG:3857) の場合
            if x_range > 1000 and y_range > 1000:  # メートル単位
                # Web Mercator から WGS84 (経度緯度) に変換
                transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
                lon_lat_coords = np.array([
                    transformer.transform(x, y) for x, y in coords
                ])

                # 境界の設定（少し余裕を持たせる）
                margin_lon = (lon_lat_coords[:, 0].max() - lon_lat_coords[:, 0].min()) * 0.1
                margin_lat = (lon_lat_coords[:, 1].max() - lon_lat_coords[:, 1].min()) * 0.1

                west = lon_lat_coords[:, 0].min() - margin_lon
                east = lon_lat_coords[:, 0].max() + margin_lon
                south = lon_lat_coords[:, 1].min() - margin_lat
                north = lon_lat_coords[:, 1].max() + margin_lat

            else:
                # 既に経度緯度の場合
                margin_x = x_range * 0.1
                margin_y = y_range * 0.1

                west = coords[:, 0].min() - margin_x
                east = coords[:, 0].max() + margin_x
                south = coords[:, 1].min() - margin_y
                north = coords[:, 1].max() + margin_y

            # contextilyで背景地図を追加
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",  # 経度緯度座標系
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=alpha,
                zoom=zoom_level
            )

            # 軸の範囲を設定
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)

            return True

        except Exception as e:
            logger.warning(f"背景地図追加エラー: {e}")
            return False

    def _convert_network_to_geoformat(self, network: nx.MultiDiGraph):
        """
        ネットワークを地理空間データ形式に変換

        Args:
            network: NetworkXグラフ

        Returns:
            (nodes_gdf, edges_gdf): GeoDataFrame形式のノード・エッジデータ
        """
        try:
            if not self.geopandas_available:
                return None, None

            # ノードのGeoDataFrame作成
            node_data = []
            for node, data in network.nodes(data=True):
                if "x" in data and "y" in data:
                    node_data.append({
                        "node_id": node,
                        "geometry": Point(data["x"], data["y"]),
                        "x": data["x"],
                        "y": data["y"]
                    })

            if not node_data:
                return None, None

            nodes_gdf = gpd.GeoDataFrame(node_data, crs="EPSG:4326")

            # エッジのGeoDataFrame作成
            edge_data = []
            for u, v, _data in network.edges(data=True):
                u_data = network.nodes[u]
                v_data = network.nodes[v]

                if all(key in u_data for key in ["x", "y"]) and all(key in v_data for key in ["x", "y"]):
                    edge_data.append({
                        "u": u,
                        "v": v,
                        "geometry": LineString([
                            (u_data["x"], u_data["y"]),
                            (v_data["x"], v_data["y"])
                        ])
                    })

            edges_gdf = gpd.GeoDataFrame(edge_data, crs="EPSG:4326") if edge_data else None

            return nodes_gdf, edges_gdf

        except Exception as e:
            logger.warning(f"地理空間データ変換エラー: {e}")
            return None, None

    def plot_network_overview(self,
                             major_network: nx.MultiDiGraph,
                             full_network: nx.MultiDiGraph | None = None,
                             results: dict[str, Any] | None = None,
                             title: str = "道路ネットワーク概要",
                             save_path: str | None = None) -> None:
        """
        ネットワーク概要の可視化

        Args:
            major_network: 主要道路ネットワーク
            full_network: 全道路ネットワーク（オプション）
            results: 分析結果（オプション）
            title: グラフタイトル
            save_path: 保存パス
        """
        try:
            import numpy as np

            if self.japanese_available:
                title_text = title
                network_labels = ["主要道路", "全道路"] if full_network else ["道路ネットワーク"]
            else:
                title_text = "Road Network Overview"
                network_labels = ["Major Roads", "All Roads"] if full_network else ["Road Network"]

            fig_width = 16 if full_network else 12
            fig, axes = plt.subplots(1, 2 if full_network else 1, figsize=(fig_width, 8))
            fig.suptitle(title_text, fontsize=16, fontweight="bold")

            # 軸を配列として統一
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            # 主要ネットワークの描画
            self._plot_single_network(major_network, axes[0], network_labels[0])

            # 全ネットワークの描画（利用可能な場合）
            if full_network and len(axes) > 1:
                self._plot_single_network(full_network, axes[1], network_labels[1])

            # 分析結果の統計表示
            if results:
                self._add_statistics_text(fig, results)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"ネットワーク概要可視化エラー: {e}")
            self._create_error_plot(str(e), save_path)

    def _plot_single_network(self, network: nx.MultiDiGraph, ax: plt.Axes,
                           network_label: str) -> None:
        """単一ネットワークの描画"""
        try:

            if network.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(network_label)
                return

            # ノード座標の取得
            pos = {}
            for node, data in network.nodes(data=True):
                if "x" in data and "y" in data:
                    pos[node] = (data["x"], data["y"])

            if not pos:
                ax.text(0.5, 0.5, "座標データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(network_label)
                return

            # エッジの描画
            nx.draw_networkx_edges(
                network, pos, ax=ax, edge_color="gray", alpha=0.6, width=0.5
            )

            # ノードの描画（高次数ノードを強調）
            degrees = dict(network.degree())
            if degrees:
                max_degree = max(degrees.values())
                node_sizes = [30 + (degrees.get(node, 0) / max_degree) * 50
                             for node in network.nodes()]

                nx.draw_networkx_nodes(
                    network, pos, ax=ax, node_color="red", node_size=node_sizes,
                    alpha=0.7
                )

            ax.set_title(f"{network_label} ({network.number_of_nodes():,}ノード)")
            ax.set_aspect("equal")
            ax.axis("off")

        except Exception as e:
            logger.warning(f"単一ネットワーク描画エラー: {e}")
            ax.text(0.5, 0.5, f"描画エラー: {str(e)[:50]}...", ha="center", va="center",
                   transform=ax.transAxes, fontsize=10)

    def _add_statistics_text(self, fig: plt.Figure, results: dict[str, Any]) -> None:
        """統計情報をテキストで追加"""
        try:
            stats_text = self._format_statistics(results)
            fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily="monospace",
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8})
        except Exception as e:
            logger.warning(f"統計テキスト追加エラー: {e}")

    def _format_statistics(self, results: dict[str, Any]) -> str:
        """統計情報のフォーマット"""
        try:
            major_network = results.get("major_network", {})

            if self.japanese_available:
                stats_lines = [
                    "=== 基本統計 ===",
                    f"ノード数: {major_network.get('node_count', 0):,}",
                    f"エッジ数: {major_network.get('edge_count', 0):,}",
                    f"α指数: {major_network.get('alpha_index', 0):.2f}",
                    f"γ指数: {major_network.get('gamma_index', 0):.2f}",
                    f"道路密度: {major_network.get('road_density', 0):.2f} km/km²"
                ]
            else:
                stats_lines = [
                    "=== Basic Statistics ===",
                    f"Nodes: {major_network.get('node_count', 0):,}",
                    f"Edges: {major_network.get('edge_count', 0):,}",
                    f"Alpha Index: {major_network.get('alpha_index', 0):.2f}",
                    f"Gamma Index: {major_network.get('gamma_index', 0):.2f}",
                    f"Road Density: {major_network.get('road_density', 0):.2f} km/km²"
                ]

            return "\n".join(stats_lines)

        except Exception as e:
            logger.warning(f"統計フォーマットエラー: {e}")
            return "統計情報の表示でエラーが発生しました"

    def plot_metrics_comparison(self,
                              results: dict[str, Any],
                              title: str = "指標比較",
                              save_path: str | None = None) -> None:
        """
        各種指標の比較可視化

        Args:
            results: 分析結果
            title: グラフタイトル
            save_path: 保存パス
        """
        try:
            if self.japanese_available:
                title_text = title
                metric_labels = ["α指数", "β指数", "γ指数", "道路密度"]
            else:
                title_text = "Metrics Comparison"
                metric_labels = ["Alpha Index", "Beta Index", "Gamma Index", "Road Density"]

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title_text, fontsize=14, fontweight="bold")

            major_network = results.get("major_network", {})

            # 基本指標の棒グラフ
            indices = [
                major_network.get("alpha_index", 0),
                major_network.get("beta_index", 0),
                major_network.get("gamma_index", 0),
                major_network.get("road_density", 0)
            ]

            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
            ax1.bar(metric_labels, indices, color=colors)
            ax1.set_title("基本指標" if self.japanese_available else "Basic Metrics")
            ax1.set_ylabel("値" if self.japanese_available else "Value")
            ax1.tick_params(axis="x", rotation=45)

            # ネットワーク規模
            sizes = [
                major_network.get("node_count", 0),
                major_network.get("edge_count", 0)
            ]
            size_labels = ["ノード", "エッジ"] if self.japanese_available else ["Nodes", "Edges"]

            ax2.bar(size_labels, sizes, color=["#FF9F43", "#10AC84"])
            ax2.set_title("ネットワーク規模" if self.japanese_available else "Network Size")
            ax2.set_ylabel("数量" if self.japanese_available else "Count")

            # 効率性指標
            efficiency_metrics = {
                ("迂回率" if self.japanese_available else "Circuity"): major_network.get("avg_circuity", 0),
                ("密度" if self.japanese_available else "Density"): major_network.get("density", 0) * 1000
            }

            ax3.bar(efficiency_metrics.keys(), efficiency_metrics.values(), color=["#EE5A24", "#0984E3"])
            ax3.set_title("効率性指標" if self.japanese_available else "Efficiency Metrics")
            ax3.set_ylabel("値" if self.japanese_available else "Value")

            # 接続性分析
            connectivity_data = {
                ("平均次数" if self.japanese_available else "Avg Degree"): major_network.get("avg_degree", 0),
                ("最大次数" if self.japanese_available else "Max Degree"): major_network.get("max_degree", 0)
            }

            ax4.bar(connectivity_data.keys(), connectivity_data.values(), color=["#00B894", "#6C5CE7"])
            ax4.set_title("接続性分析" if self.japanese_available else "Connectivity Analysis")
            ax4.set_ylabel("次数" if self.japanese_available else "Degree")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"指標比較可視化エラー: {e}")
            self._create_error_plot(str(e), save_path)

    def save_network_graph(self,
                          network: nx.MultiDiGraph,
                          save_path: str,
                          title: str = "道路ネットワーク",
                          network_type: str = "major",
                          figsize: tuple[float, float] = (12, 10),
                          dpi: int = 300,
                          show_stats: bool = True,
                          edge_color: str = "red",
                          node_color: str = "red",
                          edge_width: float = 1.0,
                          node_size_range: tuple[int, int] = (20, 100),
                          show_basemap: bool = True,
                          basemap_alpha: float = 0.6) -> bool:
        """
        道路ネットワークグラフを図として保存（背景地図付き）

        Args:
            network: 保存対象のネットワーク
            save_path: 保存先パス（拡張子: .png, .jpg, .pdf, .svg等）
            title: 図のタイトル
            network_type: ネットワークタイプ（"major", "full", "custom"）
            figsize: 図のサイズ (幅, 高さ)
            dpi: 解像度
            show_stats: 統計情報の表示有無
            edge_color: エッジの色
            node_color: ノードの色
            edge_width: エッジの太さ
            node_size_range: ノードサイズの範囲 (最小, 最大)
            show_basemap: 背景地図の表示有無
            basemap_alpha: 背景地図の透過率

        Returns:
            保存成功時True
        """
        try:
            from pathlib import Path

            import numpy as np

            logger.info(f"ネットワークグラフ保存開始: {save_path}")

            # 保存先ディレクトリの作成
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            if network.number_of_nodes() == 0:
                logger.warning("空のネットワークです")
                return False

            # 図の初期化
            fig, ax = plt.subplots(figsize=figsize)

            # タイトル設定
            if self.japanese_available:
                title_text = f"{title} ({network.number_of_nodes():,}ノード, {network.number_of_edges():,}エッジ)"
            else:
                title_text = f"{title} ({network.number_of_nodes():,} nodes, {network.number_of_edges():,} edges)"

            ax.set_title(title_text, fontsize=14, fontweight="bold", pad=20)

            # ノード座標の取得
            pos = {}
            for node, data in network.nodes(data=True):
                if "x" in data and "y" in data:
                    pos[node] = (data["x"], data["y"])

            if not pos:
                ax.text(0.5, 0.5, "座標データがありません", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12)
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                plt.close()
                return False

            # 背景地図の追加
            if show_basemap:
                basemap_success = self._add_basemap(ax, network, alpha=basemap_alpha)
                if not basemap_success:
                    logger.info("背景地図なしで描画を続行します")

            # エッジの描画
            nx.draw_networkx_edges(
                network, pos, ax=ax,
                edge_color=edge_color,
                alpha=0.8,
                width=edge_width
            )

            # ノードサイズの計算（次数に基づく）
            degrees = dict(network.degree())
            if degrees:
                max_degree = max(degrees.values())
                min_degree = min(degrees.values())

                if max_degree > min_degree:
                    # 次数に応じてノードサイズを調整
                    node_sizes = []
                    for node in network.nodes():
                        degree = degrees.get(node, 0)
                        normalized_degree = (degree - min_degree) / (max_degree - min_degree)
                        size = node_size_range[0] + normalized_degree * (node_size_range[1] - node_size_range[0])
                        node_sizes.append(size)
                else:
                    node_sizes = [node_size_range[0]] * len(network.nodes())
            else:
                node_sizes = [node_size_range[0]] * len(network.nodes())

            # ノードの描画
            nx.draw_networkx_nodes(
                network, pos, ax=ax,
                node_color=node_color,
                node_size=node_sizes,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5
            )

            # 統計情報の表示
            if show_stats:
                self._add_network_stats_text(ax, network, network_type)

            # 軸の設定
            ax.set_aspect("equal")
            if not show_basemap or not self.contextily_available:
                ax.axis("off")

            # グリッドと境界の調整（背景地図がない場合）
            if not show_basemap and pos:
                coords = np.array(list(pos.values()))
                x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
                y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.05

                ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
                ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

            # 保存
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"ネットワークグラフ保存完了: {save_path}")
            return True

        except Exception as e:
            logger.error(f"ネットワークグラフ保存エラー: {e}")
            return False

    def save_network_comparison(self,
                              major_network: nx.MultiDiGraph,
                              full_network: nx.MultiDiGraph,
                              save_path: str,
                              title: str = "道路ネットワーク比較",
                              figsize: tuple[float, float] = (16, 8),
                              dpi: int = 300) -> bool:
        """
        主要道路と全道路ネットワークの比較図を保存

        Args:
            major_network: 主要道路ネットワーク
            full_network: 全道路ネットワーク
            save_path: 保存先パス
            title: 図のタイトル
            figsize: 図のサイズ
            dpi: 解像度

        Returns:
            保存成功時True
        """
        try:
            from pathlib import Path

            logger.info(f"ネットワーク比較図保存開始: {save_path}")

            # 保存先ディレクトリの作成
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            if self.japanese_available:
                fig.suptitle(title, fontsize=16, fontweight="bold")
                labels = ["主要道路ネットワーク", "全道路ネットワーク"]
            else:
                fig.suptitle("Road Network Comparison", fontsize=16, fontweight="bold")
                labels = ["Major Road Network", "Full Road Network"]

            networks = [major_network, full_network]
            colors = ["red", "blue"]

            for _i, (network, ax, label, color) in enumerate(zip(networks, [ax1, ax2], labels, colors, strict=False)):
                if network.number_of_nodes() == 0:
                    ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(label)
                    continue

                # ノード座標の取得
                pos = {}
                for node, data in network.nodes(data=True):
                    if "x" in data and "y" in data:
                        pos[node] = (data["x"], data["y"])

                if pos:
                    # ネットワークの描画
                    nx.draw_networkx_edges(
                        network, pos, ax=ax,
                        edge_color="gray",
                        alpha=0.4,
                        width=0.3
                    )

                    # 次数に基づくノードサイズ
                    degrees = dict(network.degree())
                    if degrees:
                        max_degree = max(degrees.values())
                        node_sizes = [20 + (degrees.get(node, 0) / max_degree) * 60
                                     for node in network.nodes()]
                    else:
                        node_sizes = [20] * len(network.nodes())

                    nx.draw_networkx_nodes(
                        network, pos, ax=ax,
                        node_color=color,
                        node_size=node_sizes,
                        alpha=0.7
                    )

                    # 統計情報の追加
                    stats_text = f"{network.number_of_nodes():,}ノード\n{network.number_of_edges():,}エッジ"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment="top",
                           bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

                ax.set_title(label, fontweight="bold")
                ax.set_aspect("equal")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"ネットワーク比較図保存完了: {save_path}")
            return True

        except Exception as e:
            logger.error(f"ネットワーク比較図保存エラー: {e}")
            return False

    def save_network_with_metrics(self,
                                 network: nx.MultiDiGraph,
                                 results: dict[str, Any],
                                 save_path: str,
                                 title: str = "道路ネットワーク + 分析結果",
                                 figsize: tuple[float, float] = (16, 10),
                                 dpi: int = 300) -> bool:
        """
        ネットワークグラフと分析結果を組み合わせた図を保存

        Args:
            network: ネットワーク
            results: 分析結果
            save_path: 保存先パス
            title: 図のタイトル
            figsize: 図のサイズ
            dpi: 解像度

        Returns:
            保存成功時True
        """
        try:
            from pathlib import Path


            logger.info(f"ネットワーク+メトリクス図保存開始: {save_path}")

            # 保存先ディレクトリの作成
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

            # メインのネットワーク図（上段全体）
            ax_main = fig.add_subplot(gs[0, :])

            if self.japanese_available:
                ax_main.set_title(title, fontsize=16, fontweight="bold")
            else:
                ax_main.set_title("Road Network with Analysis Results", fontsize=16, fontweight="bold")

            # ネットワークの描画
            if network.number_of_nodes() > 0:
                pos = {}
                for node, data in network.nodes(data=True):
                    if "x" in data and "y" in data:
                        pos[node] = (data["x"], data["y"])

                if pos:
                    # エッジの描画
                    nx.draw_networkx_edges(
                        network, pos, ax=ax_main,
                        edge_color="gray",
                        alpha=0.6,
                        width=0.5
                    )

                    # ノードの描画（次数による色分け）
                    degrees = dict(network.degree())
                    if degrees:
                        max_degree = max(degrees.values())
                        node_colors = [degrees.get(node, 0) for node in network.nodes()]
                        node_sizes = [30 + (degrees.get(node, 0) / max_degree) * 70
                                     for node in network.nodes()]

                        scatter = ax_main.scatter(
                            [pos[node][0] for node in network.nodes()],
                            [pos[node][1] for node in network.nodes()],
                            c=node_colors,
                            s=node_sizes,
                            cmap="Reds",
                            alpha=0.7
                        )

                        # カラーバー
                        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
                        cbar.set_label("ノード次数" if self.japanese_available else "Node Degree")

            ax_main.set_aspect("equal")
            ax_main.axis("off")

            # 分析結果のサブプロット（下段）
            major_network = results.get("major_network", {})

            # 基本指標
            ax1 = fig.add_subplot(gs[1, 0])
            metrics = ["α指数", "β指数", "γ指数"] if self.japanese_available else ["Alpha", "Beta", "Gamma"]
            values = [
                major_network.get("alpha_index", 0),
                major_network.get("beta_index", 0),
                major_network.get("gamma_index", 0)
            ]
            ax1.bar(metrics, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
            ax1.set_title("基本指標" if self.japanese_available else "Basic Metrics")
            ax1.tick_params(axis="x", rotation=45)

            # ネットワーク規模
            ax2 = fig.add_subplot(gs[1, 1])
            size_labels = ["ノード", "エッジ"] if self.japanese_available else ["Nodes", "Edges"]
            size_values = [
                major_network.get("node_count", 0),
                major_network.get("edge_count", 0)
            ]
            ax2.bar(size_labels, size_values, color=["#FF9F43", "#10AC84"])
            ax2.set_title("規模" if self.japanese_available else "Scale")

            # 効率性指標
            ax3 = fig.add_subplot(gs[1, 2])
            eff_labels = ["道路密度", "迂回率"] if self.japanese_available else ["Road Density", "Circuity"]
            eff_values = [
                major_network.get("road_density", 0),
                major_network.get("avg_circuity", 0)
            ]
            ax3.bar(eff_labels, eff_values, color=["#EE5A24", "#0984E3"])
            ax3.set_title("効率性" if self.japanese_available else "Efficiency")
            ax3.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"ネットワーク+メトリクス図保存完了: {save_path}")
            return True

        except Exception as e:
            logger.error(f"ネットワーク+メトリクス図保存エラー: {e}")
            return False

    def save_axial_analysis(self,
                           axial_results: dict[str, Any],
                           save_path: str,
                           title: str = "軸線分析結果",
                           figsize: tuple[float, float] = (16, 12),
                           dpi: int = 300) -> bool:
        """
        軸線分析結果を図として保存

        Args:
            axial_results: 軸線分析結果
            save_path: 保存先パス
            title: 図のタイトル
            figsize: 図のサイズ
            dpi: 解像度

        Returns:
            保存成功時True
        """
        try:
            from pathlib import Path


            logger.info(f"軸線分析図保存開始: {save_path}")

            # 保存先ディレクトリの作成
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # 軸線マップとIntegration値の取得
            axial_map = axial_results.get("axial_map")
            integration_values = axial_results.get("integration_values", {})
            network_metrics = axial_results.get("network_metrics", {})
            integration_stats = axial_results.get("integration_statistics", {})

            if not axial_map or axial_map.number_of_nodes() == 0:
                logger.warning("軸線マップデータがありません")
                return False

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

            if self.japanese_available:
                fig.suptitle(title, fontsize=16, fontweight="bold")
            else:
                fig.suptitle("Axial Analysis Results", fontsize=16, fontweight="bold")

            # 1. メイン軸線マップ（上段全体）
            ax_main = fig.add_subplot(gs[0, :])
            self._plot_axial_map(ax_main, axial_map, integration_values)

            # 2. Integration値ヒストグラム（下段左）
            ax1 = fig.add_subplot(gs[1, 0])
            self._plot_integration_histogram(ax1, integration_values, integration_stats)

            # 3. ネットワーク指標（下段中）
            ax2 = fig.add_subplot(gs[1, 1])
            self._plot_axial_network_metrics(ax2, network_metrics)

            # 4. Integration統計（下段右）
            ax3 = fig.add_subplot(gs[1, 2])
            self._plot_integration_statistics(ax3, integration_stats)

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"軸線分析図保存完了: {save_path}")
            return True

        except Exception as e:
            logger.error(f"軸線分析図保存エラー: {e}")
            return False

    def save_axial_lines_only(self,
                             axial_results: dict[str, Any],
                             save_path: str,
                             title: str = "軸線マップ",
                             figsize: tuple[float, float] = (12, 10),
                             dpi: int = 300,
                             show_integration: bool = True,
                             show_basemap: bool = True,
                             basemap_alpha: float = 0.6) -> bool:
        """
        軸線のみを強調した図を保存（背景地図付き）

        Args:
            axial_results: 軸線分析結果
            save_path: 保存先パス
            title: 図のタイトル
            figsize: 図のサイズ
            dpi: 解像度
            show_integration: Integration値による色分け表示
            show_basemap: 背景地図の表示有無
            basemap_alpha: 背景地図の透過率

        Returns:
            保存成功時True
        """
        try:
            from pathlib import Path

            import numpy as np

            logger.info(f"軸線マップ保存開始: {save_path}")

            # 保存先ディレクトリの作成
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            axial_map = axial_results.get("axial_map")
            integration_values = axial_results.get("integration_values", {})

            if not axial_map or axial_map.number_of_nodes() == 0:
                logger.warning("軸線マップデータがありません")
                return False

            fig, ax = plt.subplots(figsize=figsize)

            if self.japanese_available:
                title_text = f"{title} ({axial_map.number_of_nodes()}軸線)"
            else:
                title_text = f"Axial Map ({axial_map.number_of_nodes()} lines)"

            ax.set_title(title_text, fontsize=14, fontweight="bold", pad=20)

            # 軸線の座標取得
            pos = {}
            for node, data in axial_map.nodes(data=True):
                if "x" in data and "y" in data:
                    pos[node] = (data["x"], data["y"])

            if not pos:
                ax.text(0.5, 0.5, "座標データがありません", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12)
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                plt.close()
                return False

            # 背景地図の追加
            if show_basemap:
                basemap_success = self._add_basemap(ax, axial_map, alpha=basemap_alpha)
                if not basemap_success:
                    logger.info("背景地図なしで軸線描画を続行します")

            if show_integration and integration_values:
                # Integration値による色分け
                node_colors = []
                for node in axial_map.nodes():
                    int_val = integration_values.get(node, 0)
                    node_colors.append(int_val)

                if node_colors:
                    # エッジの描画（軽く）
                    nx.draw_networkx_edges(
                        axial_map, pos, ax=ax,
                        edge_color="gray",
                        alpha=0.3,
                        width=2.0
                    )

                    # ノードの描画（Integration値による色分け）
                    scatter = ax.scatter(
                        [pos[node][0] for node in axial_map.nodes()],
                        [pos[node][1] for node in axial_map.nodes()],
                        c=node_colors,
                        cmap="plasma",
                        s=80,
                        alpha=0.9,
                        edgecolors="white",
                        linewidth=1.0
                    )

                    # カラーバー
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label("Integration Value" if not self.japanese_available else "統合値")

                    # 統計情報の追加
                    if integration_values:
                        mean_int = np.mean(list(integration_values.values()))
                        std_int = np.std(list(integration_values.values()))

                        stats_text = (f"軸線数: {len(integration_values)}\n"
                                     f"平均統合値: {mean_int:.3f}\n"
                                     f"標準偏差: {std_int:.3f}" if self.japanese_available else
                                     f"Lines: {len(integration_values)}\n"
                                     f"Mean Int.V: {mean_int:.3f}\n"
                                     f"Std Dev: {std_int:.3f}")

                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               verticalalignment="top",
                               bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
            else:
                # 単純な軸線表示
                nx.draw_networkx_edges(
                    axial_map, pos, ax=ax,
                    edge_color="red",
                    alpha=0.8,
                    width=2.5
                )

                nx.draw_networkx_nodes(
                    axial_map, pos, ax=ax,
                    node_color="darkred",
                    node_size=60,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=1.0
                )

            ax.set_aspect("equal")
            if not show_basemap or not self.contextily_available:
                ax.axis("off")

            # 境界調整（背景地図がない場合）
            if not show_basemap and pos:
                coords = np.array(list(pos.values()))
                x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
                y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.05

                ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
                ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"軸線マップ保存完了: {save_path}")
            return True

        except Exception as e:
            logger.error(f"軸線マップ保存エラー: {e}")
            return False

    def _plot_axial_map(self, ax: plt.Axes, axial_map: nx.Graph,
                       integration_values: dict[Any, float]) -> None:
        """軸線マップの描画"""
        try:

            if self.japanese_available:
                ax.set_title(f"軸線マップ ({axial_map.number_of_nodes()}軸線)", fontweight="bold")
            else:
                ax.set_title(f"Axial Map ({axial_map.number_of_nodes()} lines)", fontweight="bold")

            # 座標の取得
            pos = {}
            for node, data in axial_map.nodes(data=True):
                if "x" in data and "y" in data:
                    pos[node] = (data["x"], data["y"])

            if not pos:
                ax.text(0.5, 0.5, "座標データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12)
                return

            if integration_values:
                # Integration値による色分け
                node_colors = [integration_values.get(node, 0) for node in axial_map.nodes()]

                # エッジの描画
                nx.draw_networkx_edges(
                    axial_map, pos, ax=ax,
                    edge_color="lightgray",
                    alpha=0.4,
                    width=0.5
                )

                # ノードの描画
                scatter = ax.scatter(
                    [pos[node][0] for node in axial_map.nodes()],
                    [pos[node][1] for node in axial_map.nodes()],
                    c=node_colors,
                    cmap="plasma",
                    s=50,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.3
                )

                # カラーバー
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
                cbar.set_label("統合値" if self.japanese_available else "Integration Value", fontsize=10)
            else:
                # 単純表示
                nx.draw(axial_map, pos, ax=ax, node_color="red", edge_color="gray",
                       node_size=30, width=0.5, alpha=0.7)

            ax.set_aspect("equal")
            ax.axis("off")

        except Exception as e:
            logger.warning(f"軸線マップ描画エラー: {e}")

    def _plot_integration_histogram(self, ax: plt.Axes, integration_values: dict[Any, float],
                                   integration_stats: dict[str, float]) -> None:
        """Integration値ヒストグラムの描画"""
        try:
            if not integration_values:
                ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=10)
                return

            values = list(integration_values.values())

            ax.hist(values, bins=20, color="skyblue", alpha=0.7, edgecolor="black")

            # 統計線の追加
            if integration_stats:
                mean_val = integration_stats.get("mean", 0)
                ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"平均: {mean_val:.3f}")
                ax.legend()

            if self.japanese_available:
                ax.set_title("統合値分布", fontweight="bold")
                ax.set_xlabel("統合値")
                ax.set_ylabel("頻度")
            else:
                ax.set_title("Integration Value Distribution", fontweight="bold")
                ax.set_xlabel("Integration Value")
                ax.set_ylabel("Frequency")

        except Exception as e:
            logger.warning(f"Integration値ヒストグラム描画エラー: {e}")

    def _plot_axial_network_metrics(self, ax: plt.Axes, network_metrics: dict[str, Any]) -> None:
        """軸線ネットワーク指標の描画"""
        try:
            if not network_metrics:
                ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=10)
                return

            if self.japanese_available:
                metrics = {
                    "軸線数": network_metrics.get("axial_lines", 0),
                    "接続数": network_metrics.get("axial_connections", 0),
                    "孤立数": network_metrics.get("axial_islands", 0)
                }
                ax.set_title("ネットワーク指標", fontweight="bold")
                ax.set_ylabel("数量")
            else:
                metrics = {
                    "Lines": network_metrics.get("axial_lines", 0),
                    "Connections": network_metrics.get("axial_connections", 0),
                    "Islands": network_metrics.get("axial_islands", 0)
                }
                ax.set_title("Network Metrics", fontweight="bold")
                ax.set_ylabel("Count")

            colors = ["#FF6B6B", "#4ECDC4", "#FFD93D"]
            bars = ax.bar(metrics.keys(), metrics.values(), color=colors)

            # 値をバーの上に表示
            for bar, value in zip(bars, metrics.values(), strict=False):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.01,
                       f"{value}", ha="center", va="bottom", fontweight="bold")

            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            logger.warning(f"軸線ネットワーク指標描画エラー: {e}")

    def _plot_integration_statistics(self, ax: plt.Axes, integration_stats: dict[str, float]) -> None:
        """Integration統計の描画"""
        try:
            if not integration_stats:
                ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                       transform=ax.transAxes, fontsize=10)
                return

            if self.japanese_available:
                stats = {
                    "平均": integration_stats.get("mean", 0),
                    "標準偏差": integration_stats.get("std", 0),
                    "最大": integration_stats.get("max", 0),
                    "最小": integration_stats.get("min", 0)
                }
                ax.set_title("統合値統計", fontweight="bold")
                ax.set_ylabel("値")
            else:
                stats = {
                    "Mean": integration_stats.get("mean", 0),
                    "Std Dev": integration_stats.get("std", 0),
                    "Max": integration_stats.get("max", 0),
                    "Min": integration_stats.get("min", 0)
                }
                ax.set_title("Integration Statistics", fontweight="bold")
                ax.set_ylabel("Value")

            colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB"]
            bars = ax.bar(stats.keys(), stats.values(), color=colors)

            # 値をバーの上に表示
            for bar, value in zip(bars, stats.values(), strict=False):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats.values())*0.01,
                       f"{value:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            logger.warning(f"Integration統計描画エラー: {e}")

    def _create_error_plot(self, error_message: str, save_path: str | None = None) -> None:
        """エラー時のフォールバック可視化"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            error_text = ("可視化エラーが発生しました" if self.japanese_available
                         else "Visualization Error Occurred")
            detail_text = f"{error_text}\n{str(error_message)[:100]}..."

            ax.text(0.5, 0.5, detail_text, ha="center", va="center",
                   transform=ax.transAxes, fontsize=10,
                   bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightcoral"})

            title_text = ("Space Syntax 分析 - エラー" if self.japanese_available
                         else "Space Syntax Analysis - Error")
            ax.set_title(title_text)
            ax.set_xticks([])
            ax.set_yticks([])

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()
            plt.close()

        except Exception as nested_e:
            logger.error(f"フォールバック可視化も失敗: {nested_e}")
            raise

    def _add_network_stats_text(self, ax: plt.Axes, network: nx.MultiDiGraph,
                               network_type: str) -> None:
        """ネットワーク統計情報をテキストで追加"""
        try:
            degrees = dict(network.degree())

            if self.japanese_available:
                stats_text = (
                    f"ノード数: {network.number_of_nodes():,}\n"
                    f"エッジ数: {network.number_of_edges():,}\n"
                    f"平均次数: {sum(degrees.values()) / len(degrees):.2f}\n"
                    f"最大次数: {max(degrees.values()) if degrees else 0}\n"
                    f"密度: {nx.density(network):.4f}"
                )
            else:
                stats_text = (
                    f"Nodes: {network.number_of_nodes():,}\n"
                    f"Edges: {network.number_of_edges():,}\n"
                    f"Avg Degree: {sum(degrees.values()) / len(degrees):.2f}\n"
                    f"Max Degree: {max(degrees.values()) if degrees else 0}\n"
                    f"Density: {nx.density(network):.4f}"
                )

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment="top", fontsize=10,
                   bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

        except Exception as e:
            logger.warning(f"統計テキスト追加エラー: {e}")

    def export_results(self, results: dict[str, Any], filepath: str,
                      format_type: str = "csv") -> bool:
        """
        分析結果をファイルに出力

        Args:
            results: 分析結果
            filepath: 出力先パス
            format_type: ファイル形式 ("csv", "excel", "json")

        Returns:
            出力成功時True
        """
        try:
            logger.info(f"結果出力開始: {filepath} ({format_type})")

            if format_type.lower() == "csv":
                df = self._results_to_dataframe(results)
                df.to_csv(filepath, index=False, encoding="utf-8-sig")

            elif format_type.lower() in ["excel", "xlsx"]:
                df = self._results_to_dataframe(results)
                df.to_excel(filepath, index=False)

            elif format_type.lower() == "json":
                import json

                # NetworkXオブジェクトを除去した結果を作成
                clean_results = {}
                for key, value in results.items():
                    if key not in ["major_network", "full_network"] and not isinstance(value, nx.MultiDiGraph):
                        clean_results[key] = value

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)

            else:
                raise ValueError(f"サポートされていないフォーマット: {format_type}") from None

            logger.info(f"結果出力完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"結果出力エラー: {e}")
            return False

    def _results_to_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """分析結果をDataFrameに変換"""
        data = []

        # メタデータ
        metadata = results.get("metadata", {})

        for network_type in ["major_network", "full_network"]:
            if results.get(network_type):
                network_data = results[network_type]
                row = {
                    "network_type": network_type,
                    "query": metadata.get("query", ""),
                    "node_count": network_data.get("node_count", 0),
                    "edge_count": network_data.get("edge_count", 0),
                    "avg_degree": network_data.get("avg_degree", 0),
                    "max_degree": network_data.get("max_degree", 0),
                    "density": network_data.get("density", 0),
                    "alpha_index": network_data.get("alpha_index", 0),
                    "beta_index": network_data.get("beta_index", 0),
                    "gamma_index": network_data.get("gamma_index", 0),
                    "road_density": network_data.get("road_density", 0),
                    "avg_circuity": network_data.get("avg_circuity", 0),
                    "is_connected": network_data.get("is_connected", False),
                    "largest_component_size": network_data.get("largest_component_size", 0),
                    "num_components": network_data.get("num_components", 0)
                }
                data.append(row)

        return pd.DataFrame(data)
