"""
拡張可視化モジュール - EnhancedVisualizer

軸線分析と可視領域分析の高度な可視化機能を提供します。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """
    拡張可視化を行うクラス

    軸線分析、可視領域分析の高度な可視化機能を提供します。
    """

    def __init__(self) -> None:
        """EnhancedVisualizerを初期化"""
        # matplotlib日本語フォント設定
        plt.rcParams["font.family"] = [
            "DejaVu Sans",
            "Hiragino Sans",
            "Yu Gothic",
            "Meiryo",
            "Takao",
            "IPAexGothic",
            "IPAPGothic",
            "VL PGothic",
            "Noto Sans CJK JP",
        ]

        # カスタムカラーマップの定義
        self.integration_cmap = LinearSegmentedColormap.from_list(
            "integration", ["blue", "cyan", "yellow", "red"]
        )

    def plot_axial_map_with_integration(self,
                                       axial_graph: nx.Graph,
                                       integration_values: dict[int, float],
                                       title: str = "軸線マップ (Integration Value)",
                                       save_path: str | None = None) -> None:
        """
        Integration Valueでカラーリングした軸線マップを表示

        Args:
            axial_graph: 軸線グラフ
            integration_values: Integration Value
            title: グラフタイトル
            save_path: 保存パス
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))

            if axial_graph.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "軸線データなし",
                       transform=ax.transAxes, ha="center", va="center")
                ax.set_title(title)
                plt.show()
                return

            # Integration Valueの正規化
            if integration_values:
                min_int = min(integration_values.values())
                max_int = max(integration_values.values())

                if max_int > min_int:
                    norm_integration = {
                        node: (value - min_int) / (max_int - min_int)
                        for node, value in integration_values.items()
                    }
                else:
                    norm_integration = dict.fromkeys(integration_values, 0.5)
            else:
                norm_integration = dict.fromkeys(axial_graph.nodes(), 0.5)

            # 軸線の描画
            for node in axial_graph.nodes():
                node_data = axial_graph.nodes[node]

                if "geometry" in node_data:
                    line_geom = node_data["geometry"]
                    coords = list(line_geom.coords)

                    if len(coords) >= 2:
                        xs, ys = zip(*coords, strict=False)

                        # Integration Valueに基づく色設定
                        color_value = norm_integration.get(node, 0.5)
                        color = self.integration_cmap(color_value)

                        # 線幅もIntegration Valueに基づく
                        line_width = 1 + 3 * color_value

                        ax.plot(xs, ys, color=color, linewidth=line_width, alpha=0.8)

            # カラーバーの追加
            if integration_values:
                sm = plt.cm.ScalarMappable(
                    cmap=self.integration_cmap,
                    norm=plt.Normalize(vmin=min_int, vmax=max_int)
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
                cbar.set_label("Integration Value", fontsize=12)

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_aspect("equal")
            ax.axis("off")

            # 統計情報の表示
            if integration_values:
                mean_int = np.mean(list(integration_values.values()))
                std_int = np.std(list(integration_values.values()))

                stats_text = f"軸線数: {axial_graph.number_of_nodes()}\n"
                stats_text += f"平均Int.V: {mean_int:.3f}\n"
                stats_text += f"標準偏差: {std_int:.3f}"

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment="top",
                       bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"軸線マップ可視化エラー: {e}")

    def plot_visibility_field(self,
                             visibility_results: dict[str, Any],
                             title: str = "可視領域フィールド分析",
                             save_path: str | None = None) -> None:
        """
        可視領域フィールドの可視化

        Args:
            visibility_results: 可視領域分析結果
            title: グラフタイトル
            save_path: 保存パス
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(title, fontsize=16, fontweight="bold")

            sampling_points = visibility_results.get("sampling_points", [])
            isovist_results = visibility_results.get("isovist_results", [])

            if not sampling_points or not isovist_results:
                for ax in axes.flat:
                    ax.text(0.5, 0.5, "データなし",
                           transform=ax.transAxes, ha="center", va="center")
                plt.show()
                return

            # 可視面積の分布
            areas = [result.get("visible_area", 0) for result in isovist_results]
            xs = [point[0] for point in sampling_points]
            ys = [point[1] for point in sampling_points]

            # 1. 可視面積のヒートマップ
            scatter = axes[0, 0].scatter(xs, ys, c=areas, cmap="viridis",
                                       s=30, alpha=0.7)
            axes[0, 0].set_title("可視面積分布")
            axes[0, 0].set_xlabel("X座標 (m)")
            axes[0, 0].set_ylabel("Y座標 (m)")
            plt.colorbar(scatter, ax=axes[0, 0], label="可視面積 (m²)")

            # 2. コンパクト性の分布
            compactness = [result.get("compactness", 0) for result in isovist_results]
            scatter2 = axes[0, 1].scatter(xs, ys, c=compactness, cmap="plasma",
                                        s=30, alpha=0.7)
            axes[0, 1].set_title("コンパクト性分布")
            axes[0, 1].set_xlabel("X座標 (m)")
            axes[0, 1].set_ylabel("Y座標 (m)")
            plt.colorbar(scatter2, ax=axes[0, 1], label="コンパクト性")

            # 3. 可視面積のヒストグラム
            axes[1, 0].hist(areas, bins=30, edgecolor="black", alpha=0.7)
            axes[1, 0].set_title("可視面積の分布")
            axes[1, 0].set_xlabel("可視面積 (m²)")
            axes[1, 0].set_ylabel("頻度")

            # 4. 統計サマリー
            axes[1, 1].axis("off")

            field_stats = visibility_results.get("field_statistics", {})
            variability_metrics = visibility_results.get("variability_metrics", {})

            stats_text = "【統計サマリー】\n"
            stats_text += f"サンプリング点数: {len(sampling_points)}\n"
            stats_text += f"平均可視面積: {field_stats.get('mean_visible_area', 0):.1f} m²\n"
            stats_text += f"可視面積標準偏差: {field_stats.get('std_visible_area', 0):.1f} m²\n"
            stats_text += f"平均コンパクト性: {field_stats.get('mean_compactness', 0):.3f}\n"
            stats_text += f"平均遮蔽性: {field_stats.get('mean_occlusivity', 0):.3f}\n\n"

            stats_text += "【変動性指標】\n"
            stats_text += f"面積変動係数: {variability_metrics.get('area_coefficient_variation', 0):.3f}\n"
            stats_text += f"多様性指標: {variability_metrics.get('spatial_diversity_index', 0):.3f}\n"

            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment="top",
                           bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8})

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"可視領域フィールド可視化エラー: {e}")

    def plot_comprehensive_analysis(self,
                                  basic_results: dict[str, Any],
                                  axial_results: dict[str, Any] | None = None,
                                  visibility_results: dict[str, Any] | None = None,
                                  location_name: str = "分析対象地域",
                                  save_path: str | None = None) -> None:
        """
        包括的分析結果の可視化

        Args:
            basic_results: 基本分析結果
            axial_results: 軸線分析結果
            visibility_results: 可視領域分析結果
            location_name: 地域名
            save_path: 保存パス
        """
        try:
            # サブプロットの数を決定
            num_plots = 2  # 基本は2つ
            if axial_results:
                num_plots += 1
            if visibility_results:
                num_plots += 1

            fig = plt.figure(figsize=(20, 5 * ((num_plots + 1) // 2)))
            fig.suptitle(f"{location_name} 包括的スペースシンタックス分析",
                        fontsize=18, fontweight="bold")

            plot_idx = 1

            # 1. 基本ネットワーク比較
            if len(basic_results) >= 2:
                ax1 = plt.subplot(2, 2, plot_idx)
                self._plot_network_metrics_radar(basic_results, ax1)
                plot_idx += 1

            # 2. 基本メトリクス比較表
            ax2 = plt.subplot(2, 2, plot_idx)
            self._plot_metrics_comparison_table(basic_results, ax2)
            plot_idx += 1

            # 3. 軸線分析結果
            if axial_results:
                ax3 = plt.subplot(2, 2, plot_idx)
                self._plot_axial_analysis_summary(axial_results, ax3)
                plot_idx += 1

            # 4. 可視領域分析結果
            if visibility_results and plot_idx <= 4:
                ax4 = plt.subplot(2, 2, plot_idx)
                self._plot_visibility_summary(visibility_results, ax4)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"包括的可視化エラー: {e}")

    def _plot_network_metrics_radar(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """ネットワークメトリクスのレーダーチャート"""
        try:
            # レーダーチャート用の指標選択
            radar_metrics = ["alpha_index", "beta_index", "gamma_index",
                           "road_density", "intersection_density"]
            metric_labels = ["α指数", "β指数", "γ指数", "道路密度", "交差点密度"]

            # データの準備
            network_types = list(results.keys())
            values = []

            for network_type in network_types:
                network_values = []
                for metric in radar_metrics:
                    value = results[network_type].get(metric, 0)
                    # 正規化（0-1の範囲に）
                    if metric in ["alpha_index", "gamma_index"]:
                        normalized_value = min(value / 100.0, 1.0)
                    elif metric == "beta_index":
                        normalized_value = min(value / 3.0, 1.0)
                    elif metric in ["road_density", "intersection_density"]:
                        # 適切な最大値で正規化（地域に応じて調整必要）
                        max_density = 1000.0 if "road" in metric else 100.0
                        normalized_value = min(value / max_density, 1.0)
                    else:
                        normalized_value = min(value, 1.0)
                    network_values.append(normalized_value)
                values.append(network_values)

            # 角度の設定
            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]  # 円を閉じる

            # レーダーチャートの描画
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            colors = ["blue", "red", "green", "orange"]

            for i, (network_type, network_values) in enumerate(zip(network_types, values, strict=False)):
                network_values += network_values[:1]  # 円を閉じる

                network_name = "主要道路" if network_type == "major_network" else "全道路"

                ax.plot(angles, network_values, "o-", linewidth=2,
                       label=network_name, color=colors[i % len(colors)])
                ax.fill(angles, network_values, alpha=0.25,
                       color=colors[i % len(colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title("ネットワーク指標比較", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)

        except Exception as e:
            logger.warning(f"レーダーチャート描画エラー: {e}")
            ax.text(0.5, 0.5, "レーダーチャート描画エラー",
                   transform=ax.transAxes, ha="center", va="center")

    def _plot_metrics_comparison_table(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """メトリクス比較表の表示"""
        try:
            ax.axis("off")

            # データの準備
            metrics_to_show = [
                ("nodes", "ノード数"),
                ("edges", "エッジ数"),
                ("mu_index", "回路指数μ"),
                ("alpha_index", "α指数(%)"),
                ("beta_index", "β指数"),
                ("gamma_index", "γ指数(%)"),
                ("avg_circuity", "平均迂回率"),
            ]

            table_data = []
            for metric_key, metric_name in metrics_to_show:
                row = [metric_name]
                for network_type in results:
                    value = results[network_type].get(metric_key, 0)
                    if isinstance(value, float):
                        row.append(f"{value:.2f}")
                    else:
                        row.append(str(value))
                table_data.append(row)

            # ヘッダーの準備
            headers = ["指標"]
            for network_type in results:
                network_name = "主要道路" if network_type == "major_network" else "全道路"
                headers.append(network_name)

            # テーブルの描画
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # ヘッダーのスタイル設定
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#4CAF50")
                table[(0, i)].set_text_props(weight="bold", color="white")

            ax.set_title("指標比較表", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"比較表描画エラー: {e}")
            ax.text(0.5, 0.5, "比較表描画エラー",
                   transform=ax.transAxes, ha="center", va="center")

    def _plot_axial_analysis_summary(self, axial_results: dict[str, Any], ax: plt.Axes) -> None:
        """軸線分析結果のサマリー表示"""
        try:
            ax.axis("off")

            network_metrics = axial_results.get("network_metrics", {})
            integration_stats = axial_results.get("integration_statistics", {})

            summary_text = "【軸線分析結果】\n\n"
            summary_text += f"軸線数: {network_metrics.get('axial_lines', 0)}\n"
            summary_text += f"軸線接続数: {network_metrics.get('axial_connections', 0)}\n"
            summary_text += f"アイランド数: {network_metrics.get('axial_islands', 0)}\n\n"

            summary_text += "【形態指標】\n"
            summary_text += f"格子度(GA): {network_metrics.get('grid_axiality', 0):.3f}\n"
            summary_text += f"循環度(AR): {network_metrics.get('axial_ringiness', 0):.3f}\n"
            summary_text += f"分節度(AA): {network_metrics.get('axial_articulation', 0):.3f}\n\n"

            summary_text += "【Integration Value統計】\n"
            summary_text += f"平均値: {integration_stats.get('mean', 0):.3f}\n"
            summary_text += f"標準偏差: {integration_stats.get('std', 0):.3f}\n"
            summary_text += f"最大値: {integration_stats.get('max', 0):.3f}\n"
            summary_text += f"最小値: {integration_stats.get('min', 0):.3f}\n"

            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8})

            ax.set_title("軸線分析サマリー", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"軸線分析サマリー描画エラー: {e}")
            ax.text(0.5, 0.5, "軸線分析サマリー描画エラー",
                   transform=ax.transAxes, ha="center", va="center")

    def _plot_visibility_summary(self, visibility_results: dict[str, Any], ax: plt.Axes) -> None:
        """可視領域分析結果のサマリー表示"""
        try:
            ax.axis("off")

            field_stats = visibility_results.get("field_statistics", {})
            variability_metrics = visibility_results.get("variability_metrics", {})

            summary_text = "【可視領域分析結果】\n\n"
            summary_text += f"サンプリング点数: {field_stats.get('total_sampling_points', 0)}\n\n"

            summary_text += "【可視面積統計】\n"
            summary_text += f"平均: {field_stats.get('mean_visible_area', 0):.1f} m²\n"
            summary_text += f"標準偏差: {field_stats.get('std_visible_area', 0):.1f} m²\n"
            summary_text += f"最大値: {field_stats.get('max_visible_area', 0):.1f} m²\n"
            summary_text += f"最小値: {field_stats.get('min_visible_area', 0):.1f} m²\n\n"

            summary_text += "【その他指標】\n"
            summary_text += f"平均コンパクト性: {field_stats.get('mean_compactness', 0):.3f}\n"
            summary_text += f"平均遮蔽性: {field_stats.get('mean_occlusivity', 0):.3f}\n\n"

            summary_text += "【変動性】\n"
            summary_text += f"面積変動係数: {variability_metrics.get('area_coefficient_variation', 0):.3f}\n"
            summary_text += f"多様性指標: {variability_metrics.get('spatial_diversity_index', 0):.3f}\n"

            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.8})

            ax.set_title("可視領域分析サマリー", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"可視領域サマリー描画エラー: {e}")
            ax.text(0.5, 0.5, "可視領域サマリー描画エラー",
                   transform=ax.transAxes, ha="center", va="center")

    def plot_integration_value_distribution(self,
                                           integration_values: dict[int, float],
                                           title: str = "Integration Value分布",
                                           save_path: str | None = None) -> None:
        """
        Integration Valueの分布を可視化

        Args:
            integration_values: Integration Value
            title: グラフタイトル
            save_path: 保存パス
        """
        try:
            if not integration_values:
                logger.warning("Integration Valueデータがありません")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(title, fontsize=14, fontweight="bold")

            values = list(integration_values.values())

            # ヒストグラム
            ax1.hist(values, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
            ax1.set_title("Integration Value ヒストグラム")
            ax1.set_xlabel("Integration Value")
            ax1.set_ylabel("頻度")
            ax1.grid(True, alpha=0.3)

            # 統計情報の追加
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax1.axvline(mean_val, color="red", linestyle="--",
                       label=f"平均: {mean_val:.3f}")
            ax1.axvline(mean_val + std_val, color="orange", linestyle="--",
                       label=f"+1σ: {mean_val + std_val:.3f}")
            ax1.axvline(mean_val - std_val, color="orange", linestyle="--",
                       label=f"-1σ: {mean_val - std_val:.3f}")
            ax1.legend()

            # ボックスプロット
            ax2.boxplot(values, patch_artist=True,
                       boxprops={"facecolor": "lightblue", "alpha": 0.7})
            ax2.set_title("Integration Value ボックスプロット")
            ax2.set_ylabel("Integration Value")
            ax2.grid(True, alpha=0.3)

            # 統計情報テキスト
            stats_text = f"軸線数: {len(values)}\n"
            stats_text += f"平均: {np.mean(values):.3f}\n"
            stats_text += f"中央値: {np.median(values):.3f}\n"
            stats_text += f"標準偏差: {np.std(values):.3f}\n"
            stats_text += f"最大値: {np.max(values):.3f}\n"
            stats_text += f"最小値: {np.min(values):.3f}"

            ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes,
                    verticalalignment="center",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"Integration Value分布可視化エラー: {e}")

    def create_comparison_dashboard(self,
                                  locations_results: dict[str, dict[str, Any]],
                                  save_path: str | None = None) -> None:
        """
        複数地域の比較ダッシュボードを作成

        Args:
            locations_results: 地域別分析結果
            save_path: 保存パス
        """
        try:
            num_locations = len(locations_results)
            if num_locations < 2:
                logger.warning("比較には2つ以上の地域が必要です")
                return

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("地域間比較ダッシュボード", fontsize=16, fontweight="bold")

            # データの準備
            location_names = list(locations_results.keys())
            comparison_data = self._prepare_comparison_data(locations_results)

            # 1. 回遊性指標比較
            self._plot_connectivity_comparison(comparison_data, location_names, axes[0, 0])

            # 2. アクセス性指標比較
            self._plot_accessibility_comparison(comparison_data, location_names, axes[0, 1])

            # 3. 迂回性指標比較
            self._plot_circuity_comparison(comparison_data, location_names, axes[0, 2])

            # 4. 総合レーダーチャート
            self._plot_comprehensive_radar(comparison_data, location_names, axes[1, 0])

            # 5. ランキング表
            self._plot_ranking_table(comparison_data, location_names, axes[1, 1])

            # 6. 相関分析
            self._plot_correlation_analysis(comparison_data, axes[1, 2])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"比較ダッシュボード作成エラー: {e}")

    def _prepare_comparison_data(self, locations_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """比較用データの準備"""
        try:
            comparison_data = []

            for location, results in locations_results.items():
                # 主要道路のデータを使用
                major_network_data = results.get("major_network", {})

                row = {
                    "location": location,
                    "nodes": major_network_data.get("nodes", 0),
                    "edges": major_network_data.get("edges", 0),
                    "mu_index": major_network_data.get("mu_index", 0),
                    "alpha_index": major_network_data.get("alpha_index", 0),
                    "beta_index": major_network_data.get("beta_index", 0),
                    "gamma_index": major_network_data.get("gamma_index", 0),
                    "road_density": major_network_data.get("road_density", 0),
                    "intersection_density": major_network_data.get("intersection_density", 0),
                    "avg_circuity": major_network_data.get("avg_circuity", 0),
                }
                comparison_data.append(row)

            return pd.DataFrame(comparison_data)

        except Exception as e:
            logger.warning(f"比較データ準備エラー: {e}")
            return pd.DataFrame()

    def _plot_connectivity_comparison(self, data: pd.DataFrame,
                                    location_names: list[str], ax: plt.Axes) -> None:
        """回遊性指標比較"""
        try:
            connectivity_metrics = ["mu_index", "alpha_index", "beta_index", "gamma_index"]

            x = np.arange(len(location_names))
            width = 0.2

            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

            for i, metric in enumerate(connectivity_metrics):
                values = data[metric].values
                ax.bar(x + i * width, values, width, label=metric.replace("_", " "),
                      color=colors[i], alpha=0.8)

            ax.set_title("回遊性指標比較")
            ax.set_xlabel("地域")
            ax.set_ylabel("指標値")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(location_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            logger.warning(f"回遊性比較描画エラー: {e}")

    def _plot_accessibility_comparison(self, data: pd.DataFrame,
                                     location_names: list[str], ax: plt.Axes) -> None:
        """アクセス性指標比較"""
        try:
            # 密度指標の比較
            road_density = data["road_density"].values
            intersection_density = data["intersection_density"].values

            x = np.arange(len(location_names))
            width = 0.35

            ax.bar(x - width/2, road_density, width, label="道路密度",
                  color="#FFB347", alpha=0.8)
            ax.bar(x + width/2, intersection_density * 10, width, label="交差点密度×10",
                  color="#98FB98", alpha=0.8)  # スケール調整

            ax.set_title("アクセス性指標比較")
            ax.set_xlabel("地域")
            ax.set_ylabel("密度 (m/ha, n/ha×10)")
            ax.set_xticks(x)
            ax.set_xticklabels(location_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            logger.warning(f"アクセス性比較描画エラー: {e}")

    def _plot_circuity_comparison(self, data: pd.DataFrame,
                                location_names: list[str], ax: plt.Axes) -> None:
        """迂回性指標比較"""
        try:
            circuity_values = data["avg_circuity"].values

            bars = ax.bar(location_names, circuity_values,
                         color="#DDA0DD", alpha=0.8, edgecolor="black")

            # 理想値（1.0）の基準線を追加
            ax.axhline(y=1.0, color="red", linestyle="--",
                      label="理想値 (直線距離)")

            ax.set_title("迂回性指標比較")
            ax.set_xlabel("地域")
            ax.set_ylabel("平均迂回率")
            ax.set_xticklabels(location_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 値をバーの上に表示
            for bar, value in zip(bars, circuity_values, strict=False):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{value:.2f}", ha="center", va="bottom")

        except Exception as e:
            logger.warning(f"迂回性比較描画エラー: {e}")

    def _plot_comprehensive_radar(self, data: pd.DataFrame,
                                location_names: list[str], ax: plt.Axes) -> None:
        """総合レーダーチャート"""
        try:
            # レーダーチャート用の指標（正規化済み）
            radar_metrics = ["alpha_index", "beta_index", "gamma_index"]
            metric_labels = ["α指数", "β指数", "γ指数"]

            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]

            ax = plt.subplot(2, 3, 4, projection="polar")

            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

            for i, location in enumerate(location_names):
                location_data = data[data["location"] == location].iloc[0]

                values = []
                for metric in radar_metrics:
                    value = location_data[metric]
                    # 正規化
                    if metric in ["alpha_index", "gamma_index"]:
                        normalized = min(value / 100.0, 1.0)
                    elif metric == "beta_index":
                        normalized = min(value / 3.0, 1.0)
                    else:
                        normalized = min(value, 1.0)
                    values.append(normalized)

                values += values[:1]  # 円を閉じる

                ax.plot(angles, values, "o-", linewidth=2,
                       label=location, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title("総合指標比較", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        except Exception as e:
            logger.warning(f"総合レーダーチャート描画エラー: {e}")

    def _plot_ranking_table(self, data: pd.DataFrame,
                          location_names: list[str], ax: plt.Axes) -> None:
        """ランキング表の表示"""
        try:
            ax.axis("off")

            # ランキング指標の選択
            ranking_metrics = [
                ("alpha_index", "α指数"),
                ("road_density", "道路密度"),
                ("avg_circuity", "迂回率"),
            ]

            # ランキングデータの準備
            ranking_data = []

            for metric_key, metric_name in ranking_metrics:
                sorted_data = data.sort_values(metric_key, ascending=False)
                top_location = sorted_data.iloc[0]["location"]
                top_value = sorted_data.iloc[0][metric_key]
                ranking_data.append([metric_name, top_location, f"{top_value:.2f}"])

            # テーブルの描画
            table = ax.table(cellText=ranking_data,
                           colLabels=["指標", "最高値地域", "値"],
                           cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # ヘッダーのスタイル
            for i in range(3):
                table[(0, i)].set_facecolor("#FF9800")
                table[(0, i)].set_text_props(weight="bold", color="white")

            ax.set_title("指標別ランキング", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"ランキング表描画エラー: {e}")

    def _plot_correlation_analysis(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """相関分析の表示"""
        try:
            # 数値指標のみ選択
            numeric_cols = ["alpha_index", "beta_index", "gamma_index",
                          "road_density", "intersection_density", "avg_circuity"]

            correlation_data = data[numeric_cols]
            correlation_matrix = correlation_data.corr()

            # ヒートマップの描画
            im = ax.imshow(correlation_matrix, cmap="coolwarm", aspect="auto",
                          vmin=-1, vmax=1)

            # 軸ラベルの設定
            metric_labels = ["α指数", "β指数", "γ指数", "道路密度", "交差点密度", "迂回率"]
            ax.set_xticks(range(len(metric_labels)))
            ax.set_yticks(range(len(metric_labels)))
            ax.set_xticklabels(metric_labels, rotation=45)
            ax.set_yticklabels(metric_labels)

            # 相関係数を文字で表示
            for i in range(len(metric_labels)):
                for j in range(len(metric_labels)):
                    corr_value = correlation_matrix.iloc[i, j]
                    ax.text(j, i, f"{corr_value:.2f}", ha="center", va="center",
                           color="white" if abs(corr_value) > 0.5 else "black",
                           fontweight="bold")

            ax.set_title("指標間相関分析")

            # カラーバーの追加
            plt.colorbar(im, ax=ax, shrink=0.8, label="相関係数")

        except Exception as e:
            logger.warning(f"相関分析描画エラー: {e}")

    def create_academic_report_visualization(self,
                                           results: dict[str, Any],
                                           location_name: str,
                                           save_path: str | None = None) -> None:
        """
        学術論文レベルの詳細可視化レポートを作成

        Args:
            results: 包括的分析結果
            location_name: 地域名
            save_path: 保存パス
        """
        try:
            fig = plt.figure(figsize=(20, 24))
            fig.suptitle(f"{location_name} スペースシンタックス分析 詳細レポート",
                        fontsize=20, fontweight="bold")

            # 1. ネットワーク概要 (上段左)
            ax1 = plt.subplot(4, 3, 1)
            self._plot_network_overview(results, ax1)

            # 2. 基本指標レーダーチャート (上段中)
            ax2 = plt.subplot(4, 3, 2, projection="polar")
            self._plot_basic_metrics_radar(results, ax2)

            # 3. 指標分布ヒストグラム (上段右)
            ax3 = plt.subplot(4, 3, 3)
            self._plot_metrics_distribution(results, ax3)

            # 4. 軸線マップ (2段目左)
            ax4 = plt.subplot(4, 3, 4)
            self._plot_axial_map_detailed(results, ax4)

            # 5. Integration Value分析 (2段目中)
            ax5 = plt.subplot(4, 3, 5)
            self._plot_integration_analysis(results, ax5)

            # 6. 可視領域分析 (2段目右)
            ax6 = plt.subplot(4, 3, 6)
            self._plot_visibility_analysis(results, ax6)

            # 7. 詳細統計表 (3段目全体)
            ax7 = plt.subplot(4, 1, 3)
            self._plot_detailed_statistics_table(results, ax7)

            # 8. 総合評価と提言 (最下段)
            ax8 = plt.subplot(4, 1, 4)
            self._plot_comprehensive_evaluation(results, location_name, ax8)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except Exception as e:
            logger.error(f"学術レポート可視化エラー: {e}")

    def _plot_network_overview(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """ネットワーク概要の表示"""
        try:
            ax.axis("off")

            major_network = results.get("major_network", {})
            full_network = results.get("full_network", {})

            overview_text = "【ネットワーク概要】\n\n"
            overview_text += "主要道路ネットワーク:\n"
            overview_text += f"  ノード数: {major_network.get('nodes', 0)}\n"
            overview_text += f"  エッジ数: {major_network.get('edges', 0)}\n"
            overview_text += f"  総延長: {major_network.get('total_length_m', 0):.0f}m\n\n"

            if full_network:
                overview_text += "全道路ネットワーク:\n"
                overview_text += f"  ノード数: {full_network.get('nodes', 0)}\n"
                overview_text += f"  エッジ数: {full_network.get('edges', 0)}\n"
                overview_text += f"  総延長: {full_network.get('total_length_m', 0):.0f}m\n\n"

            overview_text += f"分析エリア面積: {major_network.get('area_ha', 0):.1f}ha"

            ax.text(0.1, 0.9, overview_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8})

            ax.set_title("ネットワーク概要", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"ネットワーク概要描画エラー: {e}")

    def _plot_basic_metrics_radar(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """基本指標レーダーチャート"""
        try:
            major_network = results.get("major_network", {})

            metrics = ["alpha_index", "beta_index", "gamma_index", "avg_circuity"]
            labels = ["α指数", "β指数", "γ指数", "迂回率"]

            values = []
            for metric in metrics:
                value = major_network.get(metric, 0)
                # 正規化
                if metric in ["alpha_index", "gamma_index"]:
                    normalized = min(value / 100.0, 1.0)
                elif metric == "beta_index":
                    normalized = min(value / 3.0, 1.0)
                elif metric == "avg_circuity":
                    normalized = min((value - 1.0) / 2.0, 1.0)  # 1.0を基準とする
                else:
                    normalized = min(value, 1.0)
                values.append(max(0.0, normalized))

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, color="blue")
            ax.fill(angles, values, alpha=0.25, color="blue")

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_title("基本指標プロファイル", fontsize=12, fontweight="bold")
            ax.grid(True)

        except Exception as e:
            logger.warning(f"基本指標レーダーチャート描画エラー: {e}")

    def _plot_metrics_distribution(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """指標分布の表示"""
        try:
            major_network = results.get("major_network", {})

            metrics_values = [
                major_network.get("alpha_index", 0),
                major_network.get("beta_index", 0) * 33.33,  # スケール調整
                major_network.get("gamma_index", 0),
                major_network.get("avg_circuity", 1.0) * 50,  # スケール調整
            ]

            metric_names = ["α指数", "β指数×33", "γ指数", "迂回率×50"]
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

            bars = ax.bar(metric_names, metrics_values, color=colors, alpha=0.8)

            # 値をバーの上に表示
            for bar, value in zip(bars, metrics_values, strict=False):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{value:.1f}", ha="center", va="bottom")

            ax.set_title("主要指標分布")
            ax.set_ylabel("指標値")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)

        except Exception as e:
            logger.warning(f"指標分布描画エラー: {e}")

    def _plot_axial_map_detailed(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """軸線マップの詳細表示"""
        try:
            ax.text(0.5, 0.5, "軸線マップ\n(実装予定)",
                   transform=ax.transAxes, ha="center", va="center",
                   fontsize=14,
                   bbox={"boxstyle": "round", "facecolor": "lightyellow"})
            ax.set_title("軸線マップ", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"軸線マップ描画エラー: {e}")

    def _plot_integration_analysis(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """Integration分析の表示"""
        try:
            ax.text(0.5, 0.5, "Integration Value分析\n(実装予定)",
                   transform=ax.transAxes, ha="center", va="center",
                   fontsize=14,
                   bbox={"boxstyle": "round", "facecolor": "lightcyan"})
            ax.set_title("Integration Value分析", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"Integration分析描画エラー: {e}")

    def _plot_visibility_analysis(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """可視領域分析の表示"""
        try:
            ax.text(0.5, 0.5, "可視領域分析\n(実装予定)",
                   transform=ax.transAxes, ha="center", va="center",
                   fontsize=14,
                   bbox={"boxstyle": "round", "facecolor": "lightgreen"})
            ax.set_title("可視領域分析", fontsize=12, fontweight="bold")

        except Exception as e:
            logger.warning(f"可視領域分析描画エラー: {e}")

    def _plot_detailed_statistics_table(self, results: dict[str, Any], ax: plt.Axes) -> None:
        """詳細統計表の表示"""
        try:
            ax.axis("off")

            # 統計表のデータ準備
            major_network = results.get("major_network", {})
            full_network = results.get("full_network", {})

            table_data = [
                ["基本統計", "", ""],
                ["ノード数", str(major_network.get("nodes", 0)),
                 str(full_network.get("nodes", 0)) if full_network else "N/A"],
                ["エッジ数", str(major_network.get("edges", 0)),
                 str(full_network.get("edges", 0)) if full_network else "N/A"],
                ["道路総延長(m)", f"{major_network.get('total_length_m', 0):.0f}",
                 f"{full_network.get('total_length_m', 0):.0f}" if full_network else "N/A"],
                ["", "", ""],
                ["回遊性指標", "", ""],
                ["回路指数μ", str(major_network.get("mu_index", 0)),
                 str(full_network.get("mu_index", 0)) if full_network else "N/A"],
                ["α指数(%)", f"{major_network.get('alpha_index', 0):.1f}",
                 f"{full_network.get('alpha_index', 0):.1f}" if full_network else "N/A"],
                ["β指数", f"{major_network.get('beta_index', 0):.2f}",
                 f"{full_network.get('beta_index', 0):.2f}" if full_network else "N/A"],
                ["γ指数(%)", f"{major_network.get('gamma_index', 0):.1f}",
                 f"{full_network.get('gamma_index', 0):.1f}" if full_network else "N/A"],
                ["", "", ""],
                ["アクセス性指標", "", ""],
                ["道路密度(m/ha)", f"{major_network.get('road_density', 0):.1f}",
                 f"{full_network.get('road_density', 0):.1f}" if full_network else "N/A"],
                ["交差点密度(n/ha)", f"{major_network.get('intersection_density', 0):.1f}",
                 f"{full_network.get('intersection_density', 0):.1f}" if full_network else "N/A"],
                ["平均最短距離(m)", f"{major_network.get('avg_shortest_path', 0):.1f}",
                 f"{full_network.get('avg_shortest_path', 0):.1f}" if full_network else "N/A"],
                ["", "", ""],
                ["迂回性指標", "", ""],
                ["平均迂回率", f"{major_network.get('avg_circuity', 0):.2f}",
                 f"{full_network.get('avg_circuity', 0):.2f}" if full_network else "N/A"],
            ]

            # ヘッダーの設定
            headers = ["指標", "主要道路", "全道路"] if full_network else ["指標", "主要道路"]

            # テーブルの描画
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc="left", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.3)

            # スタイル設定
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#2196F3")
                table[(0, i)].set_text_props(weight="bold", color="white")

            # カテゴリヘッダーのスタイル
            category_rows = [1, 6, 13, 17]  # 基本統計、回遊性、アクセス性、迂回性
            for row in category_rows:
                for col in range(len(headers)):
                    table[(row, col)].set_facecolor("#E3F2FD")
                    table[(row, col)].set_text_props(weight="bold")

            ax.set_title("詳細統計表", fontsize=14, fontweight="bold")

        except Exception as e:
            logger.warning(f"詳細統計表描画エラー: {e}")

    def _plot_comprehensive_evaluation(self, results: dict[str, Any],
                                     location_name: str, ax: plt.Axes) -> None:
        """総合評価と提言の表示"""
        try:
            ax.axis("off")

            major_network = results.get("major_network", {})

            # 評価の自動生成
            evaluation = self._generate_automatic_evaluation(major_network)

            evaluation_text = f"【{location_name} 総合評価】\n\n"
            evaluation_text += f"ネットワーク分類: {evaluation['network_type']}\n\n"
            evaluation_text += f"回遊性評価: {evaluation['connectivity_evaluation']}\n"
            evaluation_text += f"アクセス性評価: {evaluation['accessibility_evaluation']}\n"
            evaluation_text += f"迂回性評価: {evaluation['circuity_evaluation']}\n\n"
            evaluation_text += "【計画提言】\n"
            evaluation_text += f"• {evaluation['recommendation1']}\n"
            evaluation_text += f"• {evaluation['recommendation2']}\n"
            evaluation_text += f"• {evaluation['recommendation3']}"

            ax.text(0.05, 0.95, evaluation_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9})

            ax.set_title("総合評価と提言", fontsize=14, fontweight="bold")

        except Exception as e:
            logger.warning(f"総合評価描画エラー: {e}")

    def _generate_automatic_evaluation(self, metrics: dict[str, Any]) -> dict[str, str]:
        """メトリクスに基づく自動評価生成"""
        try:
            alpha = metrics.get("alpha_index", 0)
            beta = metrics.get("beta_index", 0)
            gamma = metrics.get("gamma_index", 0)
            circuity = metrics.get("avg_circuity", 1.0)
            road_density = metrics.get("road_density", 0)

            # ネットワーク分類
            if alpha > 30 and gamma > 60:
                network_type = "格子型（高回遊性）"
            elif beta > 1.5 and alpha > 20:
                network_type = "放射型（中心集約）"
            elif alpha < 10 and beta < 1.2:
                network_type = "樹状型（低回遊性）"
            else:
                network_type = "不定型（混合）"

            # 回遊性評価
            if alpha > 40:
                connectivity_eval = "優秀 - 非常に高い回遊性を持つ"
            elif alpha > 25:
                connectivity_eval = "良好 - 適度な回遊性を持つ"
            elif alpha > 15:
                connectivity_eval = "普通 - 基本的な回遊性を持つ"
            else:
                connectivity_eval = "改善必要 - 回遊性が低い"

            # アクセス性評価
            if road_density > 500:
                accessibility_eval = "優秀 - 高密度で良好なアクセス"
            elif road_density > 300:
                accessibility_eval = "良好 - 適度なアクセス性"
            elif road_density > 150:
                accessibility_eval = "普通 - 基本的なアクセス性"
            else:
                accessibility_eval = "改善必要 - アクセス性が低い"

            # 迂回性評価
            if circuity < 1.2:
                circuity_eval = "優秀 - 直線的で効率的"
            elif circuity < 1.5:
                circuity_eval = "良好 - 適度な迂回性"
            elif circuity < 2.0:
                circuity_eval = "普通 - やや迂回が多い"
            else:
                circuity_eval = "改善必要 - 迂回が多すぎる"

            # 提言生成
            recommendations = []

            if alpha < 20:
                recommendations.append("街路の接続性向上により回遊性を改善")
            if road_density < 200:
                recommendations.append("道路密度の向上によりアクセス性を向上")
            if circuity > 1.8:
                recommendations.append("直線的なルートの整備により移動効率を改善")

            # デフォルト提言
            while len(recommendations) < 3:
                default_recommendations = [
                    "歩行者優先の街路環境整備",
                    "交通結節点の機能強化",
                    "緑化による街路景観の向上"
                ]
                for rec in default_recommendations:
                    if rec not in recommendations:
                        recommendations.append(rec)
                        break

            return {
                "network_type": network_type,
                "connectivity_evaluation": connectivity_eval,
                "accessibility_evaluation": accessibility_eval,
                "circuity_evaluation": circuity_eval,
                "recommendation1": recommendations[0] if len(recommendations) > 0 else "",
                "recommendation2": recommendations[1] if len(recommendations) > 1 else "",
                "recommendation3": recommendations[2] if len(recommendations) > 2 else "",
            }

        except Exception as e:
            logger.warning(f"自動評価生成エラー: {e}")
            return {
                "network_type": "不明",
                "connectivity_evaluation": "評価不可",
                "accessibility_evaluation": "評価不可",
                "circuity_evaluation": "評価不可",
                "recommendation1": "詳細分析が必要",
                "recommendation2": "専門家による評価を推奨",
                "recommendation3": "追加データ収集を検討",
            }
