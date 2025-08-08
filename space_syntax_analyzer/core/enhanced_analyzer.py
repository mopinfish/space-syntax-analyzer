"""
拡張メインアナライザー - EnhancedSpaceSyntaxAnalyzer

軸線分析と可視領域分析を統合したメインアナライザークラス
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from .analyzer import SpaceSyntaxAnalyzer
from .axial import AxialAnalyzer
from .enhanced_visualization import EnhancedVisualizer
from .visibility import VisibilityAnalyzer

if TYPE_CHECKING:
    from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class EnhancedSpaceSyntaxAnalyzer(SpaceSyntaxAnalyzer):
    """
    拡張スペースシンタックスアナライザー

    基本分析に加えて軸線分析と可視領域分析を統合したクラス
    """

    def __init__(
        self,
        width_threshold: float = 4.0,
        network_type: str = "drive",
        crs: str = "EPSG:4326",
        enable_axial_analysis: bool = True,
        enable_visibility_analysis: bool = True,
        visibility_radius: float = 100.0,
    ) -> None:
        """
        EnhancedSpaceSyntaxAnalyzerを初期化

        Args:
            width_threshold: 道路幅員の閾値（メートル）
            network_type: OSMnxのネットワークタイプ
            crs: 座標参照系
            enable_axial_analysis: 軸線分析を有効にするか
            enable_visibility_analysis: 可視領域分析を有効にするか
            visibility_radius: 可視範囲（メートル）
        """
        # 基底クラスの初期化
        super().__init__(width_threshold, network_type, crs)

        # 拡張機能の設定
        self.enable_axial_analysis = enable_axial_analysis
        self.enable_visibility_analysis = enable_visibility_analysis
        self.visibility_radius = visibility_radius

        # 拡張アナライザーの初期化
        if self.enable_axial_analysis:
            self.axial_analyzer = AxialAnalyzer()

        if self.enable_visibility_analysis:
            self.visibility_analyzer = VisibilityAnalyzer(
                visibility_radius=visibility_radius
            )

        # 拡張可視化機能
        self.enhanced_visualizer = EnhancedVisualizer()

    def analyze_comprehensive(
        self,
        location: str | tuple[float, float, float, float] | Polygon,
        return_networks: bool = False,
        analysis_level: str = "global",
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, nx.Graph]]:
        """
        包括的分析を実行（基本分析 + 軸線分析 + 可視領域分析）

        Args:
            location: 分析対象の場所
            return_networks: ネットワークも返すかどうか
            analysis_level: 分析レベル（"global", "local", "both"）

        Returns:
            包括的分析結果、またはネットワークを含むタプル
        """
        try:
            logger.info(f"包括的分析開始: {location}")

            # ネットワーク取得
            major_network, full_network = self.get_network(location, "both")

            # 面積計算
            area_ha = self.network_manager.calculate_area_ha(major_network)

            # 基本分析
            basic_results = self.analyze(major_network, full_network, area_ha)

            # 結果辞書の初期化
            comprehensive_results = {
                "location": str(location),
                "area_ha": area_ha,
                "basic_analysis": basic_results,
            }

            # 軸線分析
            if self.enable_axial_analysis:
                logger.info("軸線分析実行中...")
                axial_results = self._perform_axial_analysis(
                    major_network, analysis_level
                )
                comprehensive_results["axial_analysis"] = axial_results

            # 可視領域分析
            if self.enable_visibility_analysis:
                logger.info("可視領域分析実行中...")
                visibility_results = self._perform_visibility_analysis(major_network)
                comprehensive_results["visibility_analysis"] = visibility_results

            # 統合評価
            comprehensive_results["integrated_evaluation"] = self._generate_integrated_evaluation(
                comprehensive_results
            )

            logger.info("包括的分析完了")

            if return_networks:
                networks = {
                    "major_network": major_network,
                    "full_network": full_network,
                }
                return comprehensive_results, networks

            return comprehensive_results

        except Exception as e:
            logger.error(f"包括的分析エラー: {e}")
            raise

    def _perform_axial_analysis(self, network: nx.Graph,
                              analysis_level: str = "global") -> dict[str, Any]:
        """
        軸線分析を実行

        Args:
            network: 道路ネットワーク
            analysis_level: 分析レベル

        Returns:
            軸線分析結果
        """
        try:
            if not self.enable_axial_analysis:
                return {}

            # 軸線分析の実行
            axial_results = self.axial_analyzer.calculate_axial_summary(network)

            # 分析レベルに応じた追加計算
            axial_map = axial_results.get("axial_map", nx.Graph())

            if analysis_level in ["global", "both"]:
                global_integration = self.axial_analyzer.analyze_global_integration(axial_map)
                axial_results["global_integration"] = global_integration

            if analysis_level in ["local", "both"]:
                local_integration = self.axial_analyzer.analyze_local_integration(axial_map)
                axial_results["local_integration"] = local_integration

            return axial_results

        except Exception as e:
            logger.warning(f"軸線分析実行エラー: {e}")
            return {}

    def _perform_visibility_analysis(self, network: nx.Graph) -> dict[str, Any]:
        """
        可視領域分析を実行

        Args:
            network: 道路ネットワーク

        Returns:
            可視領域分析結果
        """
        try:
            if not self.enable_visibility_analysis:
                return {}

            # 可視領域フィールド分析
            visibility_field = self.visibility_analyzer.analyze_visibility_field(
                network, sampling_distance=25.0
            )

            # 視覚的接続性分析
            visual_connectivity = self.visibility_analyzer.analyze_visual_connectivity(network)

            # VGA分析
            visibility_graph = self.visibility_analyzer.calculate_visibility_graph(network)

            return {
                "visibility_field": visibility_field,
                "visual_connectivity": visual_connectivity,
                "visibility_graph": visibility_graph,
            }

        except Exception as e:
            logger.warning(f"可視領域分析実行エラー: {e}")
            return {}

    def _generate_integrated_evaluation(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        統合評価を生成

        Args:
            results: 包括的分析結果

        Returns:
            統合評価結果
        """
        try:
            basic_analysis = results.get("basic_analysis", {})
            axial_analysis = results.get("axial_analysis", {})
            visibility_analysis = results.get("visibility_analysis", {})

            # 基本メトリクスから総合スコア計算
            major_network = basic_analysis.get("major_network", {})

            # 回遊性スコア（0-100）
            alpha = major_network.get("alpha_index", 0)
            gamma = major_network.get("gamma_index", 0)
            connectivity_score = (alpha + gamma) / 2

            # アクセス性スコア（0-100）
            road_density = major_network.get("road_density", 0)
            intersection_density = major_network.get("intersection_density", 0)
            # 正規化（地域特性に応じて調整必要）
            density_score = min((road_density / 10 + intersection_density * 5), 100)

            # 効率性スコア（0-100）
            circuity = major_network.get("avg_circuity", 1.0)
            efficiency_score = max(0, (2.0 - circuity) / 1.0 * 100)

            # 総合スコア
            overall_score = (connectivity_score + density_score + efficiency_score) / 3

            # 評価レベルの判定
            if overall_score >= 80:
                evaluation_level = "A - 優秀"
            elif overall_score >= 65:
                evaluation_level = "B - 良好"
            elif overall_score >= 50:
                evaluation_level = "C - 普通"
            elif overall_score >= 35:
                evaluation_level = "D - 要改善"
            else:
                evaluation_level = "E - 大幅改善必要"

            # 軸線分析の統合
            axial_integration = {}
            if axial_analysis:
                integration_stats = axial_analysis.get("integration_statistics", {})
                if integration_stats:
                    axial_integration = {
                        "mean_integration": integration_stats.get("mean", 0),
                        "integration_variability": integration_stats.get("std", 0),
                    }

            # 可視領域分析の統合
            visibility_integration = {}
            if visibility_analysis:
                field_stats = visibility_analysis.get("visibility_field", {}).get("field_statistics", {})
                if field_stats:
                    visibility_integration = {
                        "mean_visible_area": field_stats.get("mean_visible_area", 0),
                        "visibility_variability": field_stats.get("std_visible_area", 0),
                    }

            return {
                "connectivity_score": connectivity_score,
                "accessibility_score": density_score,
                "efficiency_score": efficiency_score,
                "overall_score": overall_score,
                "evaluation_level": evaluation_level,
                "axial_integration": axial_integration,
                "visibility_integration": visibility_integration,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"統合評価生成エラー: {e}")
            return {
                "connectivity_score": 0,
                "accessibility_score": 0,
                "efficiency_score": 0,
                "overall_score": 0,
                "evaluation_level": "評価不可",
                "axial_integration": {},
                "visibility_integration": {},
            }

    def compare_locations(self,
                         locations: list[str | tuple[float, float, float, float] | Polygon],
                         location_names: list[str] | None = None) -> dict[str, Any]:
        """
        複数地域の比較分析

        Args:
            locations: 比較対象地域のリスト
            location_names: 地域名のリスト

        Returns:
            比較分析結果
        """
        try:
            logger.info(f"複数地域比較分析開始: {len(locations)}地域")

            if location_names is None:
                location_names = [f"地域{i+1}" for i in range(len(locations))]

            if len(location_names) != len(locations):
                raise ValueError("地域数と地域名の数が一致しません")

            # 各地域の分析
            locations_results = {}

            for location, name in zip(locations, location_names, strict=False):
                logger.info(f"地域分析中: {name}")

                try:
                    result = self.analyze_comprehensive(location)
                    locations_results[name] = result
                except Exception as e:
                    logger.warning(f"地域 {name} の分析でエラー: {e}")
                    continue

            if not locations_results:
                raise ValueError("有効な分析結果が得られませんでした")

            # 比較分析の実行
            comparison_results = self._perform_comparative_analysis(locations_results)

            return {
                "locations_results": locations_results,
                "comparison_analysis": comparison_results,
                "location_names": location_names,
            }

        except Exception as e:
            logger.error(f"複数地域比較分析エラー: {e}")
            raise

    def _perform_comparative_analysis(self,
                                    locations_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        比較分析を実行

        Args:
            locations_results: 地域別分析結果

        Returns:
            比較分析結果
        """
        try:
            # 基本指標の比較
            basic_comparison = self._compare_basic_metrics(locations_results)

            # ランキング生成
            rankings = self._generate_rankings(locations_results)

            # 類似性分析
            similarity_analysis = self._analyze_similarity(locations_results)

            # 特徴的地域の特定
            characteristic_locations = self._identify_characteristic_locations(locations_results)

            return {
                "basic_comparison": basic_comparison,
                "rankings": rankings,
                "similarity_analysis": similarity_analysis,
                "characteristic_locations": characteristic_locations,
            }

        except Exception as e:
            logger.warning(f"比較分析実行エラー: {e}")
            return {}

    def _compare_basic_metrics(self, locations_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """基本指標の比較"""
        try:
            comparison_data = []

            for location_name, results in locations_results.items():
                basic_analysis = results.get("basic_analysis", {})
                major_network = basic_analysis.get("major_network", {})

                row = {
                    "location": location_name,
                    "nodes": major_network.get("nodes", 0),
                    "edges": major_network.get("edges", 0),
                    "area_ha": major_network.get("area_ha", 0),
                    "mu_index": major_network.get("mu_index", 0),
                    "alpha_index": major_network.get("alpha_index", 0),
                    "beta_index": major_network.get("beta_index", 0),
                    "gamma_index": major_network.get("gamma_index", 0),
                    "road_density": major_network.get("road_density", 0),
                    "intersection_density": major_network.get("intersection_density", 0),
                    "avg_circuity": major_network.get("avg_circuity", 0),
                    "overall_score": results.get("integrated_evaluation", {}).get("overall_score", 0),
                }
                comparison_data.append(row)

            return pd.DataFrame(comparison_data)

        except Exception as e:
            logger.warning(f"基本指標比較エラー: {e}")
            return pd.DataFrame()

    def _generate_rankings(self, locations_results: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
        """各指標のランキングを生成"""
        try:
            comparison_df = self._compare_basic_metrics(locations_results)

            if comparison_df.empty:
                return {}

            ranking_metrics = [
                "alpha_index", "road_density", "intersection_density",
                "overall_score"
            ]

            rankings = {}

            for metric in ranking_metrics:
                if metric in comparison_df.columns:
                    sorted_data = comparison_df.sort_values(metric, ascending=False)
                    ranking = [
                        (row["location"], row[metric])
                        for _, row in sorted_data.iterrows()
                    ]
                    rankings[metric] = ranking

            return rankings

        except Exception as e:
            logger.warning(f"ランキング生成エラー: {e}")
            return {}

    def _analyze_similarity(self, locations_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """地域間の類似性分析"""
        try:
            comparison_df = self._compare_basic_metrics(locations_results)

            if comparison_df.empty or len(comparison_df) < 2:
                return {}

            # 数値指標のみ選択
            numeric_cols = ["alpha_index", "beta_index", "gamma_index",
                          "road_density", "intersection_density", "avg_circuity"]

            numeric_data = comparison_df[numeric_cols]

            # 正規化
            normalized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
            normalized_data = normalized_data.fillna(0)

            # 距離行列の計算
            from scipy.spatial.distance import pdist, squareform

            distances = pdist(normalized_data.values, metric="euclidean")
            distance_matrix = squareform(distances)

            # 最も類似している地域ペアを特定
            location_names = comparison_df["location"].tolist()
            min_distance_idx = np.unravel_index(
                np.argmin(distance_matrix + np.eye(len(location_names)) * 1000),
                distance_matrix.shape
            )

            most_similar_pair = (
                location_names[min_distance_idx[0]],
                location_names[min_distance_idx[1]]
            )

            # クラスタリング分析
            similarity_clusters = self._perform_clustering(normalized_data, location_names)

            return {
                "distance_matrix": distance_matrix.tolist(),
                "location_names": location_names,
                "most_similar_pair": most_similar_pair,
                "similarity_clusters": similarity_clusters,
            }

        except Exception as e:
            logger.warning(f"類似性分析エラー: {e}")
            return {}

    def _perform_clustering(self, data: pd.DataFrame,
                          location_names: list[str]) -> dict[str, list[str]]:
        """クラスタリング分析"""
        try:
            from sklearn.cluster import KMeans

            # 適切なクラスター数を決定（2-4の範囲）
            n_clusters = min(4, max(2, len(location_names) // 2))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)

            # クラスターごとに地域を分類
            clusters = {}
            for i in range(n_clusters):
                cluster_locations = [
                    location_names[j] for j, label in enumerate(cluster_labels)
                    if label == i
                ]
                clusters[f"クラスター{i+1}"] = cluster_locations

            return clusters

        except ImportError:
            logger.warning("scikit-learnが利用できません。クラスタリング分析をスキップ")
            return {}
        except Exception as e:
            logger.warning(f"クラスタリング分析エラー: {e}")
            return {}

    def _identify_characteristic_locations(self,
                                         locations_results: dict[str, dict[str, Any]]) -> dict[str, str]:
        """特徴的地域の特定"""
        try:
            comparison_df = self._compare_basic_metrics(locations_results)

            if comparison_df.empty:
                return {}

            characteristics = {}

            # 最高回遊性地域
            max_alpha_idx = comparison_df["alpha_index"].idxmax()
            characteristics["最高回遊性"] = comparison_df.loc[max_alpha_idx, "location"]

            # 最高密度地域
            max_density_idx = comparison_df["road_density"].idxmax()
            characteristics["最高道路密度"] = comparison_df.loc[max_density_idx, "location"]

            # 最高効率地域（最低迂回率）
            min_circuity_idx = comparison_df["avg_circuity"].idxmin()
            characteristics["最高移動効率"] = comparison_df.loc[min_circuity_idx, "location"]

            # 最もバランスの取れた地域
            # 各指標を正規化して分散が最小の地域を選択
            normalized_metrics = comparison_df[["alpha_index", "road_density", "avg_circuity"]].copy()
            for col in normalized_metrics.columns:
                normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / \
                                        (normalized_metrics[col].max() - normalized_metrics[col].min())

            normalized_metrics["variance"] = normalized_metrics.var(axis=1)
            min_variance_idx = normalized_metrics["variance"].idxmin()
            characteristics["最もバランス良好"] = comparison_df.loc[min_variance_idx, "location"]

            return characteristics

        except Exception as e:
            logger.warning(f"特徴的地域特定エラー: {e}")
            return {}

    def visualize_comprehensive(self,
                              results: dict[str, Any],
                              save_path: str | None = None) -> None:
        """
        包括的分析結果の可視化

        Args:
            results: 包括的分析結果
            save_path: 保存パス
        """
        try:
            location_name = results.get("location", "分析対象地域")
            basic_results = results.get("basic_analysis", {})
            axial_results = results.get("axial_analysis")
            visibility_results = results.get("visibility_analysis")

            self.enhanced_visualizer.plot_comprehensive_analysis(
                basic_results, axial_results, visibility_results,
                location_name, save_path
            )

        except Exception as e:
            logger.error(f"包括的可視化エラー: {e}")

    def create_comparison_dashboard(self,
                                  comparison_results: dict[str, Any],
                                  save_path: str | None = None) -> None:
        """
        比較ダッシュボードを作成

        Args:
            comparison_results: 比較分析結果
            save_path: 保存パス
        """
        try:
            locations_results = comparison_results.get("locations_results", {})

            self.enhanced_visualizer.create_comparison_dashboard(
                locations_results, save_path
            )

        except Exception as e:
            logger.error(f"比較ダッシュボード作成エラー: {e}")

    def generate_academic_report(self,
                               results: dict[str, Any],
                               output_path: str,
                               include_visualizations: bool = True) -> str:
        """
        学術論文レベルの詳細レポートを生成

        Args:
            results: 包括的分析結果
            output_path: 出力パス
            include_visualizations: 可視化を含むかどうか

        Returns:
            生成されたレポートのパス
        """
        try:
            location_name = results.get("location", "分析対象地域")

            # レポート本文の生成
            report_content = self._generate_detailed_report_content(results, location_name)

            # ファイル出力
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            # 可視化の保存
            if include_visualizations:
                vis_path = output_path.replace(".txt", "_visualizations.png")
                self.enhanced_visualizer.create_academic_report_visualization(
                    results, location_name, vis_path
                )

            logger.info(f"学術レポート生成完了: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"学術レポート生成エラー: {e}")
            raise

    def _generate_detailed_report_content(self, results: dict[str, Any],
                                        location_name: str) -> str:
        """詳細レポート内容の生成"""
        try:
            basic_analysis = results.get("basic_analysis", {})
            axial_analysis = results.get("axial_analysis", {})
            visibility_analysis = results.get("visibility_analysis", {})
            integrated_evaluation = results.get("integrated_evaluation", {})

            major_network = basic_analysis.get("major_network", {})
            full_network = basic_analysis.get("full_network", {})

            report = f"""# {location_name} スペースシンタックス分析 詳細レポート

## 1. 分析概要

本レポートは、スペースシンタックス理論に基づく {location_name} の都市空間分析結果を示す。
分析対象エリア面積: {results.get('area_ha', 0):.1f}ヘクタール
分析実行日時: {integrated_evaluation.get('analysis_timestamp', 'N/A')}

## 2. 基本ネットワーク分析結果

### 2.1 主要道路ネットワーク（幅員4m以上）

- ノード数: {major_network.get('nodes', 0)}
- エッジ数: {major_network.get('edges', 0)}
- 道路総延長: {major_network.get('total_length_m', 0):.0f}m

#### 回遊性指標
- 回路指数（μ）: {major_network.get('mu_index', 0)}
- α指数: {major_network.get('alpha_index', 0):.1f}%
- β指数: {major_network.get('beta_index', 0):.2f}
- γ指数: {major_network.get('gamma_index', 0):.1f}%

#### アクセス性指標
- 平均最短距離: {major_network.get('avg_shortest_path', 0):.1f}m
- 道路密度: {major_network.get('road_density', 0):.1f}m/ha
- 交差点密度: {major_network.get('intersection_density', 0):.1f}n/ha

#### 迂回性指標
- 平均迂回率: {major_network.get('avg_circuity', 0):.2f}

"""

            # 全道路ネットワークの情報（存在する場合）
            if full_network:
                report += f"""
### 2.2 全道路ネットワーク

- ノード数: {full_network.get('nodes', 0)}
- エッジ数: {full_network.get('edges', 0)}
- 道路総延長: {full_network.get('total_length_m', 0):.0f}m

#### 主要道路との比較
- α指数変化: {full_network.get('alpha_index', 0) - major_network.get('alpha_index', 0):.1f}%
- 道路密度変化: {full_network.get('road_density', 0) - major_network.get('road_density', 0):.1f}m/ha
- 迂回率変化: {full_network.get('avg_circuity', 0) - major_network.get('avg_circuity', 0):.2f}

"""

            # 軸線分析結果
            if axial_analysis:
                network_metrics = axial_analysis.get("network_metrics", {})
                integration_stats = axial_analysis.get("integration_statistics", {})

                report += f"""
## 3. 軸線分析結果

### 3.1 軸線ネットワーク基本統計
- 軸線数: {network_metrics.get('axial_lines', 0)}
- 軸線接続数: {network_metrics.get('axial_connections', 0)}
- アイランド数: {network_metrics.get('axial_islands', 0)}

### 3.2 形態指標
- 格子度（GA）: {network_metrics.get('grid_axiality', 0):.3f}
- 循環度（AR）: {network_metrics.get('axial_ringiness', 0):.3f}
- 分節度（AA）: {network_metrics.get('axial_articulation', 0):.3f}

### 3.3 Integration Value統計
- 平均値: {integration_stats.get('mean', 0):.3f}
- 標準偏差: {integration_stats.get('std', 0):.3f}
- 最大値: {integration_stats.get('max', 0):.3f}
- 最小値: {integration_stats.get('min', 0):.3f}

"""

            # 可視領域分析結果
            if visibility_analysis:
                field_stats = visibility_analysis.get("visibility_field", {}).get("field_statistics", {})

                if field_stats:
                    report += f"""
## 4. 可視領域分析結果

### 4.1 可視領域統計
- サンプリング点数: {field_stats.get('total_sampling_points', 0)}
- 平均可視面積: {field_stats.get('mean_visible_area', 0):.1f}m²
- 可視面積標準偏差: {field_stats.get('std_visible_area', 0):.1f}m²
- 最大可視面積: {field_stats.get('max_visible_area', 0):.1f}m²
- 最小可視面積: {field_stats.get('min_visible_area', 0):.1f}m²

### 4.2 視認性指標
- 平均コンパクト性: {field_stats.get('mean_compactness', 0):.3f}
- 平均遮蔽性: {field_stats.get('mean_occlusivity', 0):.3f}

"""

            # 統合評価
            report += f"""
## 5. 統合評価

### 5.1 スコア評価
- 回遊性スコア: {integrated_evaluation.get('connectivity_score', 0):.1f}/100
- アクセス性スコア: {integrated_evaluation.get('accessibility_score', 0):.1f}/100
- 効率性スコア: {integrated_evaluation.get('efficiency_score', 0):.1f}/100
- 総合スコア: {integrated_evaluation.get('overall_score', 0):.1f}/100

### 5.2 評価レベル
{integrated_evaluation.get('evaluation_level', '評価不可')}

## 6. 結論と提言

{location_name}の都市空間構造は、スペースシンタックス理論に基づく分析により以下の特徴が明らかになった：

1. 回遊性: {self._get_connectivity_conclusion(major_network)}
2. アクセス性: {self._get_accessibility_conclusion(major_network)}
3. 効率性: {self._get_efficiency_conclusion(major_network)}

### 6.1 計画提言
今後の都市計画において以下の点を考慮することを提言する：

- 街路ネットワークの接続性向上による回遊性の改善
- 適切な道路密度の維持によるアクセス性の確保
- 直線的なルートの整備による移動効率の向上

---
*本レポートは space-syntax-analyzer v0.1.0 により自動生成されました*
"""

            return report

        except Exception as e:
            logger.warning(f"詳細レポート内容生成エラー: {e}")
            return f"# {location_name} 分析レポート\n\nレポート生成中にエラーが発生しました。"

    def _get_connectivity_conclusion(self, metrics: dict[str, Any]) -> str:
        """回遊性の結論を生成"""
        alpha = metrics.get("alpha_index", 0)
        if alpha > 30:
            return "高い回遊性を持ち、移動の選択肢が豊富"
        elif alpha > 15:
            return "適度な回遊性を持つが、改善の余地あり"
        else:
            return "回遊性が低く、街路の接続性向上が必要"

    def _get_accessibility_conclusion(self, metrics: dict[str, Any]) -> str:
        """アクセス性の結論を生成"""
        density = metrics.get("road_density", 0)
        if density > 400:
            return "高い道路密度により良好なアクセス性を確保"
        elif density > 200:
            return "適度なアクセス性を持つが、局所的改善が有効"
        else:
            return "アクセス性が低く、道路インフラの充実が必要"

    def _get_efficiency_conclusion(self, metrics: dict[str, Any]) -> str:
        """効率性の結論を生成"""
        circuity = metrics.get("avg_circuity", 1.0)
        if circuity < 1.3:
            return "効率的な移動が可能な直線的なネットワーク"
        elif circuity < 1.7:
            return "適度な迂回性で、景観との両立が可能"
        else:
            return "迂回が多く、移動効率の改善が望ましい"
