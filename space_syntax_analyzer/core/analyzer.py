"""
Space Syntax分析器 (OSMnx v2.0対応版)

このモジュールは道路ネットワークのSpace Syntax分析を実行します。
"""

import logging
from typing import Any

import networkx as nx
import pandas as pd

from .metrics import SpaceSyntaxMetrics
from .network import NetworkManager
from .visualization import NetworkVisualizer

logger = logging.getLogger(__name__)


class SpaceSyntaxAnalyzer:
    """Space Syntax分析メインクラス (OSMnx v2.0対応)"""

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

        logger.info(f"SpaceSyntaxAnalyzer初期化完了: {network_type}, {width_threshold}m")

    def analyze_place(self, place_query: str | tuple[float, float, float, float],
                     return_networks: bool = False) -> dict[str, Any] | tuple[dict[str, Any], tuple[nx.MultiDiGraph | None, nx.MultiDiGraph | None]]:
        """
        地域のSpace Syntax分析を実行

        Args:
            place_query: 地名または(left, bottom, right, top)のbbox
            return_networks: ネットワークオブジェクトも返すかどうか

        Returns:
            分析結果辞書、return_networks=Trueの場合は(結果, (主要ネットワーク, 全ネットワーク))
        """
        try:
            logger.info(f"分析開始: {place_query}")

            # ネットワーク取得
            major_net, full_net = self._get_networks(place_query)

            if major_net is None and full_net is None:
                logger.error("ネットワーク取得に完全に失敗しました")
                return self._create_empty_result(return_networks, "ネットワーク取得失敗")

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
                "analysis_status": "success"
            }

            logger.info("分析完了")

            if return_networks:
                return results, (major_net, full_net)
            else:
                return results

        except Exception as e:
            logger.error(f"分析中にエラー発生: {e}")
            return self._create_empty_result(return_networks, str(e))

    def _get_networks(self, place_query: str | tuple[float, float, float, float]) -> tuple[nx.MultiDiGraph | None, nx.MultiDiGraph | None]:
        """
        主要道路と全道路のネットワークを取得

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
                return None, None

            if base_net is None:
                logger.error("ベースネットワーク取得に失敗")
                return None, None

            # 全道路ネットワーク（コピー）
            full_net = base_net.copy()

            # 主要道路ネットワーク（フィルタリング）
            major_net = self.network_manager.filter_major_roads(base_net.copy())

            logger.info(f"ネットワーク取得完了 - 主要: {len(major_net.nodes) if major_net else 0} ノード, "
                       f"全体: {len(full_net.nodes) if full_net else 0} ノード")

            return major_net, full_net

        except Exception as e:
            logger.error(f"ネットワーク取得エラー: {e}")
            return None, None

    def _analyze_network(self, G: nx.MultiDiGraph, network_name: str) -> dict[str, Any]:
        """
        個別ネットワークの分析を実行

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

    def _get_empty_metrics(self) -> dict[str, float]:
        """空の指標辞書を返す"""
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

    def _create_empty_result(self, return_networks: bool,
                           error_msg: str) -> dict[str, Any] | tuple[dict[str, Any], tuple[None, None]]:
        """
        空の分析結果を作成

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
                    f"  クエリ: {metadata.get('query', 'N/A')}",
                    f"  ネットワークタイプ: {metadata.get('network_type', 'N/A')}",
                    f"  主要道路幅閾値: {metadata.get('width_threshold', 'N/A')}m",
                    f"  分析ステータス: {metadata.get('analysis_status', 'N/A')}",
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
            "",
            "  Space Syntax指標:",
            f"    α指数: {network_data.get('alpha_index', 0):.1f}%",
            f"    β指数: {network_data.get('beta_index', 0):.2f}",
            f"    γ指数: {network_data.get('gamma_index', 0):.2f}",
            f"    平均迂回率: {network_data.get('avg_circuity', 1):.2f}",
            f"    道路密度: {network_data.get('road_density', 0):.1f}",
            ""
        ]
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
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            else:
                raise ValueError(f"サポートされていないフォーマット: {format_type}")

            logger.info(f"結果出力完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"結果出力エラー: {e}")
            return False

    def _results_to_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """分析結果をDataFrameに変換"""
        data = []

        for network_type in ["major_network", "full_network"]:
            if results.get(network_type):
                row = {
                    "network_type": network_type,
                    **results[network_type]
                }
                data.append(row)

        return pd.DataFrame(data)


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
                # 実際のメソッド名 plot_network_comparison を使用
                self.visualizer.plot_network_comparison(
                    major_net, full_net, results, save_path
                )
                return True
            except Exception as e:
                logger.error(f"可視化エラー: {e}")
                return False


# 便利関数
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
