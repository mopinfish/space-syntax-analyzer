# space_syntax_analyzer/core/visualization.py (Ruffエラー修正版)

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
        logger.info("NetworkVisualizer初期化完了")

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
