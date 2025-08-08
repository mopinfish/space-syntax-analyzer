# space_syntax_analyzer/core/analyzer.py（完全修正版 - 例外チェーン対応）
"""
Space Syntax分析器 (テスト互換性対応版)

既存のテストケースとの互換性を保ちつつ、堅牢化機能を提供
"""

import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ..exceptions.errors import AnalysisError, NetworkRetrievalError
from .metrics import SpaceSyntaxMetrics
from .network import NetworkManager
from .visualization import NetworkVisualizer

logger = logging.getLogger(__name__)


class SpaceSyntaxAnalyzer:
    """Space Syntax分析メインクラス (テスト互換性対応版)"""

    def __init__(self, network_type: str = "drive", width_threshold: float = 6.0):
        """
        Space Syntax分析器を初期化

        Args:
            network_type: ネットワークタイプ ("drive", "walk", "bike", "all", "all_public")
            width_threshold: 主要道路判定の幅閾値（メートル）
        """
        self.network_type = network_type
        self.width_threshold = width_threshold

        # コンポーネントの初期化
        self.network_manager = NetworkManager(network_type, width_threshold)
        self.metrics = SpaceSyntaxMetrics()
        self.visualizer = NetworkVisualizer()

        logger.info("Space Syntax Analyzer v0.1.1 初期化完了")
        logger.info(f"SpaceSyntaxAnalyzer初期化完了: {network_type}, {width_threshold}m")

    def analyze_place(self, place_query: str | tuple[float, float, float, float],
                     analysis_types: list[str] = None,
                     return_networks: bool = False) -> dict[str, Any] | tuple[dict[str, Any], tuple[nx.MultiDiGraph | None, nx.MultiDiGraph | None]]:
        """
        地域のSpace Syntax分析を実行（堅牢版）

        Args:
            place_query: 地名または(left, bottom, right, top)のbbox
            analysis_types: 実行する分析タイプ ["basic", "connectivity", "choice"]
            return_networks: ネットワークオブジェクトも返すかどうか

        Returns:
            分析結果辞書、return_networks=Trueの場合は(結果, (主要ネットワーク, 全ネットワーク))
        """
        if analysis_types is None:
            analysis_types = ["basic", "connectivity"]  # choiceは重い処理なので除外

        try:
            logger.info(f"分析開始: {place_query}")

            # ネットワーク取得（堅牢な方式）
            major_net, full_net = self._get_networks(place_query)

            if major_net is None and full_net is None:
                error_msg = f"ネットワーク取得に失敗しました: {place_query}"
                logger.error(error_msg)
                return self._create_empty_result(return_networks, error_msg)

            # 分析実行
            results = {}

            # 主要道路ネットワーク分析
            if major_net is not None:
                logger.info("主要道路ネットワーク分析中...")
                results["major_network"] = self._analyze_network(major_net, "主要道路")
            else:
                results["major_network"] = None

            # 全道路ネットワーク分析
            if full_net is not None:
                logger.info("全道路ネットワーク分析中...")
                results["full_network"] = self._analyze_network(full_net, "全道路")
            else:
                results["full_network"] = None

            # メタデータ追加
            results["metadata"] = {
                "query": str(place_query),
                "network_type": self.network_type,
                "width_threshold": self.width_threshold,
                "has_major_network": major_net is not None,
                "has_full_network": full_net is not None,
                "analysis_status": "success",
                "analysis_types": analysis_types,
                "analysis_timestamp": time.time()
            }

            # 面積計算
            if major_net is not None:
                results["area_ha"] = self.network_manager.calculate_area_ha(major_net)
            elif full_net is not None:
                results["area_ha"] = self.network_manager.calculate_area_ha(full_net)
            else:
                results["area_ha"] = 0.0

            # 統合評価
            results["integration_summary"] = self._calculate_integration_summary(results)

            logger.info("分析完了")

            if return_networks:
                return results, (major_net, full_net)
            else:
                return results

        except NetworkRetrievalError as e:
            logger.error(f"ネットワーク取得エラー: {e}")
            return self._create_empty_result(return_networks, str(e))
        except AnalysisError as e:
            logger.error(f"分析エラー: {e}")
            return self._create_empty_result(return_networks, str(e))
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return self._create_empty_result(return_networks, str(e))

    def analyze_point(self, lat: float, lon: float, radius: int = 1000,
                     analysis_types: list[str] = None) -> dict[str, Any]:
        """
        座標を指定してSpace Syntax分析を実行

        Parameters
        ----------
        lat, lon : float
            分析対象座標
        radius : int
            分析半径（メートル）
        analysis_types : list, optional
            実行する分析タイプ

        Returns
        -------
        dict
            分析結果
        """
        if analysis_types is None:
            analysis_types = ["basic", "connectivity"]

        location_str = f"({lat:.6f}, {lon:.6f})"
        logger.info(f"座標分析開始: {location_str}, 半径: {radius}m")

        try:
            # ネットワーク取得
            major_network = self.network_manager.get_network_from_point((lat, lon), radius)

            if major_network is None:
                error_msg = f"座標 {location_str} からのネットワーク取得に失敗"
                return self._create_error_result(location_str, error_msg, "coordinate_analysis_error")

            # 面積計算
            area_ha = self.network_manager.calculate_area_ha(major_network)

            # 分析実行
            results = self._analyze_network(major_network, "座標ベース")

            # 基本情報を追加
            results["location"] = location_str
            results["coordinates"] = {"lat": lat, "lon": lon, "radius": radius}
            results["analysis_timestamp"] = time.time()
            results["area_ha"] = area_ha
            results["network_stats"] = self.network_manager.get_network_stats(major_network)

            logger.info(f"座標分析完了: {location_str}")
            return results

        except Exception as e:
            logger.error(f"座標分析エラー ({location_str}): {e}")
            return self._create_error_result(location_str, str(e), "coordinate_analysis_error")

    def _get_networks(self, place_query: str | tuple[float, float, float, float]) -> tuple[nx.MultiDiGraph | None, nx.MultiDiGraph | None]:
        """
        堅牢なネットワーク取得（テスト互換メソッド名）

        Args:
            place_query: 地名またはbbox

        Returns:
            (主要道路ネットワーク, 全道路ネットワーク)
        """
        try:
            base_net = None

            # クエリの種類に応じて取得方法を変える
            if isinstance(place_query, str):
                logger.info(f"地名 '{place_query}' からネットワーク取得")
                base_net = self.network_manager.get_network_from_place(place_query, simplify=False)

            elif isinstance(place_query, tuple | list) and len(place_query) == 4:
                logger.info(f"bbox {place_query} からネットワーク取得")
                base_net = self.network_manager.get_network_from_bbox(place_query, simplify=False)

            else:
                logger.error(f"不正なクエリ形式: {place_query}")
                raise NetworkRetrievalError(f"不正なクエリ形式: {place_query}") from None

            if base_net is None:
                logger.error("ベースネットワーク取得に失敗")
                raise NetworkRetrievalError("ベースネットワーク取得に失敗") from None

            # ネットワーク品質検証
            if not self._validate_network_quality(base_net):
                logger.warning("取得したネットワークの品質が低い")
                # 品質が低くても続行（警告のみ）

            # 全道路ネットワーク（コピー）
            full_net = base_net.copy()

            # 主要道路ネットワーク（フィルタリング）
            major_net = self.network_manager.filter_major_roads(base_net.copy())

            logger.info(f"ネットワーク取得完了 - 主要: {len(major_net.nodes) if major_net else 0} ノード, "
                       f"全体: {len(full_net.nodes) if full_net else 0} ノード")

            return major_net, full_net

        except NetworkRetrievalError:
            raise
        except Exception as e:
            logger.error(f"ネットワーク取得エラー: {e}")
            raise NetworkRetrievalError(f"ネットワーク取得エラー: {e}") from e

    def _validate_network_quality(self, G: nx.MultiDiGraph) -> bool:
        """
        ネットワーク品質の検証

        Parameters
        ----------
        G : nx.MultiDiGraph
            検証対象ネットワーク

        Returns
        -------
        bool
            品質が合格かどうか
        """
        if not G or G.number_of_nodes() < 5:
            logger.warning("ネットワークが小さすぎます")
            return False

        if not nx.is_weakly_connected(G):
            logger.warning("ネットワークが接続されていません")
            # 最大連結成分を取得
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            if len(largest_cc) < G.number_of_nodes() * 0.5:  # 50%以下なら品質低
                return False

        # エッジ密度チェック
        if G.number_of_nodes() > 0:
            density = G.number_of_edges() / G.number_of_nodes()
            if density < 1.0:  # 極端に疎なネットワーク
                logger.warning(f"ネットワーク密度が低い: {density:.2f}")
                return False

        return True

    def _analyze_network(self, G: nx.MultiDiGraph, network_name: str) -> dict[str, Any]:
        """
        個別ネットワークの分析を実行（テスト互換メソッド名）

        Args:
            G: 分析対象のネットワーク
            network_name: ネットワーク名

        Returns:
            分析結果辞書
        """
        try:
            results = {
                "network_name": network_name,
                "analysis_status": "success"
            }

            # 基本統計
            basic_stats = self.network_manager.calculate_network_stats(G)
            results.update(basic_stats)

            # Space Syntax指標
            if len(G.nodes) > 0:
                syntax_metrics = self.metrics.calculate_all_metrics(G)
                results.update(syntax_metrics)
            else:
                logger.warning(f"{network_name}: ノードが0のため、Space Syntax指標をスキップ")
                results.update(self._get_empty_metrics())

            logger.info(f"{network_name} 分析完了: {results.get('node_count', 0)} ノード")
            return results

        except Exception as e:
            logger.error(f"ネットワーク分析エラー ({network_name}): {e}")
            return {
                "network_name": network_name,
                "analysis_status": "error",
                "error_message": str(e),
                **self._get_empty_metrics()
            }

    def _calculate_integration_summary(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """
        統合評価の計算

        Parameters
        ----------
        analysis_results : dict
            各分析結果

        Returns
        -------
        dict
            統合評価結果
        """
        summary = {}

        try:
            # 主要道路ネットワークの結果を使用
            network_data = analysis_results.get("major_network")
            if not network_data or network_data.get("analysis_status") != "success":
                # 主要道路がない場合は全道路を使用
                network_data = analysis_results.get("full_network")

            if not network_data or network_data.get("analysis_status") != "success":
                return {"error": "有効な分析結果がありません"}

            # ネットワーク効率性スコア
            node_count = network_data.get("node_count", 0)
            edge_count = network_data.get("edge_count", 0)
            density = network_data.get("density", 0)
            is_connected = network_data.get("is_connected", False)

            efficiency_score = 0
            if is_connected:
                efficiency_score += 30  # 基本連結性
            efficiency_score += min(density * 1000, 35)  # 密度スコア（最大35点）

            if node_count > 0:
                avg_degree = edge_count * 2 / node_count  # 無向グラフとして計算
                efficiency_score += min(avg_degree * 5, 35)  # 次数スコア（最大35点）

            summary["network_efficiency_score"] = min(efficiency_score, 100)

            # 総合評価
            summary["overall_integration_score"] = summary["network_efficiency_score"]

            # 評価レベル
            overall_score = summary["overall_integration_score"]
            if overall_score >= 80:
                summary["integration_level"] = "非常に良好"
            elif overall_score >= 60:
                summary["integration_level"] = "良好"
            elif overall_score >= 40:
                summary["integration_level"] = "普通"
            elif overall_score >= 20:
                summary["integration_level"] = "改善が必要"
            else:
                summary["integration_level"] = "大幅改善が必要"

        except Exception as e:
            logger.warning(f"統合評価計算エラー: {e}")
            summary["error"] = str(e)

        return summary

    def _get_empty_metrics(self) -> dict[str, float]:
        """空の指標辞書を返す（テスト互換メソッド名）"""
        return {
            "node_count": 0,
            "edge_count": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "min_degree": 0,
            "density": 0.0,
            "alpha_index": 0.0,
            "beta_index": 0.0,
            "gamma_index": 0.0,
            "avg_circuity": 1.0,
            "road_density": 0.0
        }

    def _create_empty_result(self, return_networks: bool, error_msg: str) -> dict[str, Any] | tuple[dict[str, Any], tuple[None, None]]:
        """
        空の分析結果を作成（テスト互換メソッド名）

        Args:
            return_networks: ネットワークも返すかどうか
            error_msg: エラーメッセージ

        Returns:
            空の結果
        """
        empty_result = {
            "major_network": None,
            "full_network": None,
            "metadata": {
                "analysis_status": "failed",
                "error_message": error_msg,
                "network_type": self.network_type,
                "width_threshold": self.width_threshold
            }
        }

        if return_networks:
            return empty_result, (None, None)
        else:
            return empty_result

    def _create_error_result(self, location: str, error_message: str, error_type: str) -> dict[str, Any]:
        """
        エラー結果の作成

        Parameters
        ----------
        location : str
            分析対象地名
        error_message : str
            エラーメッセージ
        error_type : str
            エラータイプ

        Returns
        -------
        dict
            エラー結果
        """
        return {
            "location": location,
            "error": True,
            "error_type": error_type,
            "error_message": error_message,
            "analysis_timestamp": time.time(),
            "suggestions": self._get_error_suggestions(error_type, location),
            "metadata": {
                "analysis_status": "failed",
                "error_type": error_type,
                "network_type": self.network_type,
                "width_threshold": self.width_threshold
            }
        }

    def _get_error_suggestions(self, error_type: str, location: str) -> list[str]:
        """
        エラータイプに応じた解決策提案

        Parameters
        ----------
        error_type : str
            エラータイプ
        location : str
            分析対象地名

        Returns
        -------
        list
            提案リスト
        """
        suggestions = []

        if error_type == "network_retrieval_error":
            suggestions.extend([
                "地名をより一般的な表記に変更してください（例: '渋谷駅' → '渋谷'）",
                "座標指定での分析をお試しください（analyze_point メソッド）",
                "より大きな地域名で指定してください（例: '渋谷区, 東京'）",
                "英語表記をお試しください（例: 'Shibuya, Tokyo'）"
            ])
        elif error_type == "analysis_error":
            suggestions.extend([
                "ネットワークサイズが大きすぎる可能性があります",
                "分析タイプを基本分析のみに限定してください",
                "より小さな領域での分析をお試しください"
            ])
        elif error_type == "coordinate_analysis_error":
            suggestions.extend([
                "座標値が正しいかご確認ください（緯度: -90~90, 経度: -180~180）",
                "分析半径を調整してください",
                "インターネット接続をご確認ください"
            ])

        return suggestions

    # 既存コードとの互換性メソッド
    def get_network(self, place_query: str | tuple,
                   network_selection: str = "both") -> nx.MultiDiGraph | tuple[nx.MultiDiGraph, nx.MultiDiGraph] | None:
        """
        ネットワークのみを取得（分析なし）

        Args:
            place_query: 地名またはbbox
            network_selection: "major", "full", "both"

        Returns:
            選択されたネットワーク
        """
        try:
            major_net, full_net = self._get_networks(place_query)

            if network_selection == "major":
                return major_net
            elif network_selection == "full":
                return full_net
            elif network_selection == "both":
                return major_net, full_net
            else:
                logger.error(f"不正なnetwork_selection: {network_selection}")
                return None

        except Exception as e:
            logger.error(f"ネットワーク取得エラー: {e}")
            if network_selection == "both":
                return None, None
            else:
                return None

    def generate_report(self, results: dict[str, Any], title: str = "Space Syntax 分析レポート") -> str:
        """
        分析結果の詳細レポートを生成

        Args:
            results: 分析結果
            title: レポートタイトル

        Returns:
            フォーマットされたレポート文字列
        """
        if results.get("error", False):
            return f"""
Space Syntax 分析エラーレポート
{'='*50}

分析対象: {results.get('location', 'Unknown')}
エラータイプ: {results.get('error_type', '不明')}
エラー内容: {results.get('error_message', '詳細不明')}

解決策の提案:
{chr(10).join(f"• {suggestion}" for suggestion in results.get('suggestions', []))}
"""

        try:
            lines = [
                f"\n{'='*60}",
                f"  {title}",
                f"{'='*60}",
                ""
            ]

            # メタデータセクション
            if "metadata" in results:
                metadata = results["metadata"]
                lines.extend([
                    "【基本情報】",
                    f"  分析対象: {metadata.get('query', 'N/A')}",
                    f"  ネットワークタイプ: {metadata.get('network_type', 'N/A')}",
                    f"  主要道路幅閾値: {metadata.get('width_threshold', 'N/A')}m",
                    f"  分析ステータス: {metadata.get('analysis_status', 'N/A')}",
                    f"  分析面積: {results.get('area_ha', 0):.1f}ha",
                    ""
                ])

            # 主要道路ネットワーク
            if results.get("major_network"):
                major = results["major_network"]
                lines.extend(self._format_network_section("主要道路ネットワーク", major))

            # 全道路ネットワーク
            if results.get("full_network"):
                full = results["full_network"]
                lines.extend(self._format_network_section("全道路ネットワーク", full))

            # 統合評価セクション
            integration = results.get("integration_summary", {})
            if integration and "overall_integration_score" in integration:
                lines.extend([
                    "【統合評価】",
                    f"  総合スコア: {integration.get('overall_integration_score', 0):.1f}/100",
                    f"  評価レベル: {integration.get('integration_level', '不明')}",
                    ""
                ])

            # 比較セクション
            if results.get("major_network") and results.get("full_network"):
                lines.extend(self._format_comparison_section(results["major_network"], results["full_network"]))

            lines.extend([
                "="*60,
                f"レポート生成完了 - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ])

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return f"レポート生成エラー: {e}"

    def _format_network_section(self, section_title: str, network_data: dict[str, Any]) -> list[str]:
        """ネットワークセクションをフォーマット"""
        lines = [
            f"【{section_title}】",
            f"  ノード数: {network_data.get('node_count', 0):,}",
            f"  エッジ数: {network_data.get('edge_count', 0):,}",
            f"  平均次数: {network_data.get('avg_degree', 0):.2f}",
            f"  ネットワーク密度: {network_data.get('density', 0):.4f}",
            f"  連結性: {'✓ 連結' if network_data.get('is_connected', False) else '✗ 非連結'}",
            ""
        ]

        # Space Syntax指標が存在する場合
        if "alpha_index" in network_data:
            lines.extend([
                "  Space Syntax指標:",
                f"    α指数: {network_data.get('alpha_index', 0):.1f}%",
                f"    β指数: {network_data.get('beta_index', 0):.2f}",
                f"    γ指数: {network_data.get('gamma_index', 0):.2f}",
                f"    平均迂回率: {network_data.get('avg_circuity', 1):.2f}",
                f"    道路密度: {network_data.get('road_density', 0):.1f}",
                ""
            ])

        return lines

    def _format_comparison_section(self, major: dict[str, Any], full: dict[str, Any]) -> list[str]:
        """比較セクションをフォーマット"""
        major_nodes = major.get("node_count", 0)
        full_nodes = full.get("node_count", 0)
        major_edges = major.get("edge_count", 0)
        full_edges = full.get("edge_count", 0)

        node_ratio = (major_nodes / full_nodes * 100) if full_nodes > 0 else 0
        edge_ratio = (major_edges / full_edges * 100) if full_edges > 0 else 0

        lines = [
            "【ネットワーク比較】",
            f"  主要道路ノード比率: {node_ratio:.1f}% ({major_nodes:,} / {full_nodes:,})",
            f"  主要道路エッジ比率: {edge_ratio:.1f}% ({major_edges:,} / {full_edges:,})",
            f"  主要道路平均次数: {major.get('avg_degree', 0):.2f}",
            f"  全道路平均次数: {full.get('avg_degree', 0):.2f}",
            ""
        ]
        return lines

    def visualize(self, major_net: nx.MultiDiGraph | None,
                  full_net: nx.MultiDiGraph | None,
                  results: dict[str, Any],
                  save_path: str | None = None) -> bool:
        """
        ネットワークと分析結果を可視化

        Args:
            major_net: 主要道路ネットワーク
            full_net: 全道路ネットワーク
            results: 分析結果
            save_path: 保存先パス

        Returns:
            可視化成功時True
        """
        try:
            if hasattr(self.visualizer, "plot_network_comparison"):
                self.visualizer.plot_network_comparison(
                    major_net, full_net, results, save_path
                )
            else:
                # 基本的な可視化のフォールバック
                self._basic_visualization(major_net, full_net, results, save_path)
            return True
        except Exception as e:
            logger.error(f"可視化エラー: {e}")
            return False

    def _basic_visualization(self, major_net: nx.MultiDiGraph | None,
                           full_net: nx.MultiDiGraph | None,
                           results: dict[str, Any],
                           save_path: str | None = None):
        """基本的な可視化（フォールバック）"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Space Syntax 分析結果: {results.get('metadata', {}).get('query', 'Unknown')}", fontsize=16)

            # プロットの実装（簡略版）
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, f"プロット {i+1}", ha="center", va="center", transform=ax.transAxes)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"可視化結果を保存: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"基本可視化エラー: {e}")

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
                    "analysis_status": network_data.get("analysis_status", ""),
                    **{k: v for k, v in network_data.items() if not isinstance(v, dict) and k != "analysis_status"}
                }
                data.append(row)

        if not data:
            # エラーケース
            data.append({
                "network_type": "error",
                "query": results.get("location", ""),
                "analysis_status": "failed",
                "error_message": results.get("error_message", ""),
                "error_type": results.get("error_type", "")
            })

        return pd.DataFrame(data)


# 便利関数（既存コードとの互換性維持）
def analyze_place_simple(place_query: str | tuple[float, float, float, float],
                        network_type: str = "drive") -> dict[str, Any]:
    """
    簡易分析関数

    Args:
        place_query: 地名またはbbox
        network_type: ネットワークタイプ

    Returns:
        分析結果
    """
    analyzer = SpaceSyntaxAnalyzer(network_type=network_type)
    return analyzer.analyze_place(place_query)
