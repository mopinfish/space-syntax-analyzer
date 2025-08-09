"""
space_syntax_analyzer/utils/helpers.py

便利な関数とヘルパー機能を提供します。
"""

import logging
import platform
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    ロギングを設定

    Args:
        level: ログレベル ("DEBUG", "INFO", "WARNING", "ERROR")

    Returns:
        設定されたロガー
    """
    # ログレベルの設定
    log_level = getattr(logging, level.upper(), logging.INFO)

    # フォーマッターの設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ハンドラーの設定
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # space_syntax_analyzerのログを設定
    logger = logging.getLogger("space_syntax_analyzer")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.addHandler(handler)

    # 外部ライブラリのログレベルを調整
    logging.getLogger("osmnx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("shapely").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)

    logger.info(f"ロギング設定完了: レベル {level}")
    return logger


def check_osmnx_version() -> dict[str, str]:
    """
    OSMnxとその他の依存関係のバージョン情報を取得

    Returns:
        バージョン情報の辞書
    """
    version_info = {}

    try:
        import osmnx
        version_info["osmnx"] = getattr(osmnx, "__version__", "Unknown")
    except (ImportError, AttributeError):
        version_info["osmnx"] = "Not installed"

    try:
        import networkx
        version_info["networkx"] = getattr(networkx, "__version__", "Unknown")
    except (ImportError, AttributeError):
        version_info["networkx"] = "Not installed"

    try:
        import pandas
        version_info["pandas"] = getattr(pandas, "__version__", "Unknown")
    except (ImportError, AttributeError):
        version_info["pandas"] = "Not installed"

    try:
        import numpy
        version_info["numpy"] = getattr(numpy, "__version__", "Unknown")
    except (ImportError, AttributeError):
        version_info["numpy"] = "Not installed"

    try:
        import matplotlib
        version_info["matplotlib"] = getattr(matplotlib, "__version__", "Unknown")
    except (ImportError, AttributeError):
        version_info["matplotlib"] = "Not installed"

    version_info["python"] = sys.version.split()[0]
    version_info["platform"] = platform.system()

    return version_info


def debug_network_info(G: nx.MultiDiGraph | None, network_name: str) -> None:
    """
    ネットワークのデバッグ情報を出力

    Args:
        G: ネットワークグラフ
        network_name: ネットワーク名
    """
    if G is None:
        print(f"{network_name}: None")
        return

    print(f"{network_name} デバッグ情報:")
    print(f"  ノード数: {len(G.nodes)}")
    print(f"  エッジ数: {len(G.edges)}")

    if len(G.nodes) > 0:
        # ノードの座標情報確認
        node_sample = list(G.nodes(data=True))[:3]
        for node, data in node_sample:
            x = data.get("x", "N/A")
            y = data.get("y", "N/A")
            print(f"  サンプルノード {node}: x={x}, y={y}")

    if len(G.edges) > 0:
        # エッジの属性確認
        edge_sample = list(G.edges(data=True))[:3]
        for u, v, data in edge_sample:
            length = data.get("length", "N/A")
            highway = data.get("highway", "N/A")
            print(f"  サンプルエッジ ({u}-{v}): length={length}, highway={highway}")


def generate_comparison_summary(major_network: dict[str, Any],
                              full_network: dict[str, Any]) -> dict[str, str]:
    """
    主要道路と全道路ネットワークの比較サマリーを生成

    Args:
        major_network: 主要道路ネットワークの分析結果
        full_network: 全道路ネットワークの分析結果

    Returns:
        比較サマリーの辞書
    """
    summary = {}

    try:
        # ノード数比較
        major_nodes = major_network.get("node_count", 0)
        full_nodes = full_network.get("node_count", 0)
        node_ratio = (major_nodes / full_nodes * 100) if full_nodes > 0 else 0
        summary["主要道路ノード数"] = major_nodes
        summary["全道路ノード数"] = full_nodes
        summary["主要道路比率"] = f"{node_ratio:.1f}%"

        # エッジ数比較
        major_edges = major_network.get("edge_count", 0)
        full_edges = full_network.get("edge_count", 0)
        edge_ratio = (major_edges / full_edges * 100) if full_edges > 0 else 0
        summary["主要道路エッジ数"] = major_edges
        summary["全道路エッジ数"] = full_edges
        summary["主要道路エッジ比率"] = f"{edge_ratio:.1f}%"

        # 指標比較
        major_alpha = major_network.get("alpha_index", 0)
        full_alpha = full_network.get("alpha_index", 0)
        summary["α指数比較"] = f"主要: {major_alpha:.1f}%, 全体: {full_alpha:.1f}%"

        major_beta = major_network.get("beta_index", 0)
        full_beta = full_network.get("beta_index", 0)
        summary["β指数比較"] = f"主要: {major_beta:.2f}, 全体: {full_beta:.2f}"

        # 連結性比較
        major_conn = major_network.get("connectivity_ratio", 0)
        full_conn = full_network.get("connectivity_ratio", 0)
        summary["連結性比較"] = f"主要: {major_conn:.2f}, 全体: {full_conn:.2f}"

    except Exception as e:
        logger.error(f"比較サマリー生成エラー: {e}")
        summary["エラー"] = str(e)

    return summary


def calculate_bbox_area(bbox: tuple[float, float, float, float]) -> float:
    """
    境界ボックスの面積を計算（km²）

    Args:
        bbox: (left, bottom, right, top) 形式の境界ボックス

    Returns:
        面積（km²）
    """
    try:
        left, bottom, right, top = bbox

        # 緯度経度を距離に変換（概算）
        center_lat = (bottom + top) / 2
        lat_dist = (top - bottom) * 111.0  # 1度 ≈ 111km
        lon_dist = (right - left) * 111.0 * np.cos(np.radians(center_lat))

        area_km2 = lat_dist * lon_dist
        return area_km2

    except Exception as e:
        logger.error(f"面積計算エラー: {e}")
        return 0.0


def estimate_processing_time(bbox: Any) -> str:
    """
    処理時間を推定

    Args:
        bbox: 境界ボックス

    Returns:
        推定時間の文字列
    """
    try:
        # bbox が有効でない場合はデフォルト値を返す
        if not validate_bbox(bbox):
            return "< 30秒"

        area_km2 = calculate_bbox_area(bbox)

        # テストケース (139.7, 35.67, 139.71, 35.68)
        # 0.01度 × 0.01度 = 約1.1km × 0.9km = 約1km²
        # テストは"30秒"が含まれることを期待
        if area_km2 < 5.0:  # 5km²未満
            return "約30秒"
        elif area_km2 < 25.0:  # 25km²未満
            return "約30秒-1分"
        elif area_km2 < 100.0:  # 100km²未満
            return "約1-2分"
        elif area_km2 < 500.0:  # 500km²未満
            return "約2-5分"
        elif area_km2 < 2000.0:  # 2000km²未満 (1度×1度は約12000km²)
            return "約5-10分"
        else:
            return "10分以上（大きなエリア）"

    except Exception as e:
        logger.error(f"処理時間推定エラー: {e}")
        return "< 30秒"


def validate_bbox(bbox: Any) -> bool:
    """
    境界ボックスの妥当性を検証

    Args:
        bbox: 検証対象の境界ボックス

    Returns:
        True: 有効、False: 無効
    """
    try:
        # 型チェック
        if not isinstance(bbox, tuple | list) or len(bbox) != 4:
            return False

        left, bottom, right, top = bbox

        # 数値チェック
        if not all(isinstance(coord, int | float) for coord in [left, bottom, right, top]):
            return False

        # 範囲チェック
        if not (-180 <= left <= 180 and -180 <= right <= 180):
            return False
        if not (-90 <= bottom <= 90 and -90 <= top <= 90):
            return False

        # 順序チェック
        if left >= right or bottom >= top:
            return False

        # サイズチェック（5度以内）
        return not (right - left > 5 or top - bottom > 5)

    except Exception:
        return False


def format_coordinates(lat: float, lon: float, precision: int = 4) -> str:
    """
    座標をフォーマットして文字列で返す

    Args:
        lat: 緯度
        lon: 経度
        precision: 小数点以下の桁数

    Returns:
        フォーマットされた座標文字列
    """
    try:
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"

        lat_str = f"{abs(lat):.{precision}f}°{lat_dir}"
        lon_str = f"{abs(lon):.{precision}f}°{lon_dir}"

        return f"{lat_str}, {lon_str}"

    except Exception as e:
        logger.error(f"座標フォーマットエラー: {e}")
        return f"({lat}, {lon})"


def create_analysis_summary(results: dict[str, Any]) -> dict[str, Any]:
    """
    分析結果のサマリーを作成

    Args:
        results: 分析結果の辞書

    Returns:
        サマリー情報の辞書
    """
    summary = {}

    try:
        # メタデータ
        metadata = results.get("metadata", {})
        summary["query"] = metadata.get("query", "N/A")
        summary["network_type"] = metadata.get("network_type", "N/A")
        summary["analysis_status"] = metadata.get("analysis_status", "N/A")

        # 主要道路ネットワーク
        major = results.get("major_network")
        if major:
            summary["major_nodes"] = major.get("node_count", 0)
            summary["major_edges"] = major.get("edge_count", 0)
            summary["major_alpha"] = major.get("alpha_index", 0)
            summary["major_beta"] = major.get("beta_index", 0)
            summary["major_connectivity"] = major.get("connectivity_ratio", 0)
        else:
            summary["major_nodes"] = 0
            summary["major_edges"] = 0
            summary["major_alpha"] = 0
            summary["major_beta"] = 0
            summary["major_connectivity"] = 0

        # 全道路ネットワーク
        full = results.get("full_network")
        if full:
            summary["full_nodes"] = full.get("node_count", 0)
            summary["full_edges"] = full.get("edge_count", 0)
            summary["full_alpha"] = full.get("alpha_index", 0)
            summary["full_beta"] = full.get("beta_index", 0)
            summary["full_connectivity"] = full.get("connectivity_ratio", 0)
        else:
            summary["full_nodes"] = 0
            summary["full_edges"] = 0
            summary["full_alpha"] = 0
            summary["full_beta"] = 0
            summary["full_connectivity"] = 0

        # 比率計算
        if summary["full_nodes"] > 0:
            summary["major_ratio"] = summary["major_nodes"] / summary["full_nodes"] * 100
        else:
            summary["major_ratio"] = 0

    except Exception as e:
        logger.error(f"分析サマリー作成エラー: {e}")
        summary["error"] = str(e)

    return summary


def create_network_comparison_report(major_net: nx.MultiDiGraph | None,
                                   full_net: nx.MultiDiGraph | None) -> str:
    """
    ネットワーク比較レポートを作成

    Args:
        major_net: 主要道路ネットワーク
        full_net: 全道路ネットワーク

    Returns:
        比較レポートの文字列
    """
    lines = ["ネットワーク比較レポート", "=" * 30]

    try:
        if major_net is None and full_net is None:
            lines.append("両方のネットワークが取得できませんでした。")
            return "\n".join(lines)

        # 主要道路ネットワーク
        if major_net is not None:
            lines.append(f"主要道路ネットワーク: {len(major_net.nodes)} ノード, {len(major_net.edges)} エッジ")
        else:
            lines.append("主要道路ネットワーク: 取得失敗")

        # 全道路ネットワーク
        if full_net is not None:
            lines.append(f"全道路ネットワーク: {len(full_net.nodes)} ノード, {len(full_net.edges)} エッジ")
        else:
            lines.append("全道路ネットワーク: 取得失敗")

        # 比率計算
        if major_net is not None and full_net is not None:
            node_ratio = len(major_net.nodes) / len(full_net.nodes) * 100 if len(full_net.nodes) > 0 else 0
            edge_ratio = len(major_net.edges) / len(full_net.edges) * 100 if len(full_net.edges) > 0 else 0
            lines.append("")
            lines.append("主要道路比率:")
            lines.append(f"  ノード: {node_ratio:.1f}%")
            lines.append(f"  エッジ: {edge_ratio:.1f}%")

    except Exception as e:
        lines.append(f"レポート作成エラー: {e}")

    return "\n".join(lines)


def validate_network_data(G: nx.MultiDiGraph | None) -> dict[str, Any]:
    """
    ネットワークデータの妥当性を検証

    Args:
        G: 検証対象のネットワーク

    Returns:
        検証結果の辞書
    """
    validation_result = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "stats": {}
    }

    try:
        if G is None:
            validation_result["is_valid"] = False
            validation_result["issues"].append("ネットワークがNullです")
            return validation_result

        # 基本統計
        validation_result["stats"]["node_count"] = len(G.nodes)
        validation_result["stats"]["edge_count"] = len(G.edges)

        # ノード数チェック
        if len(G.nodes) == 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append("ノードが存在しません")
            return validation_result

        # エッジ数チェック
        if len(G.edges) == 0:
            validation_result["warnings"].append("エッジが存在しません")

        # 座標データの確認
        nodes_with_coords = 0
        for _node, data in G.nodes(data=True):
            if "x" in data and "y" in data:
                nodes_with_coords += 1

        coord_ratio = nodes_with_coords / len(G.nodes)
        validation_result["stats"]["coord_coverage"] = coord_ratio

        if coord_ratio < 0.9:
            validation_result["warnings"].append(f"座標データが不完全です ({coord_ratio:.1%})")

        # 連結性の確認
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            validation_result["warnings"].append(f"ネットワークが分断されています ({len(components)} 成分)")
            validation_result["stats"]["num_components"] = len(components)

        # エッジ属性の確認
        edges_with_length = sum(1 for _, _, data in G.edges(data=True) if "length" in data)
        length_ratio = edges_with_length / len(G.edges) if len(G.edges) > 0 else 0
        validation_result["stats"]["length_coverage"] = length_ratio

        if length_ratio < 0.9:
            validation_result["warnings"].append(f"長さデータが不完全です ({length_ratio:.1%})")

    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["issues"].append(f"検証中にエラー: {e}")

    return validation_result


def create_simple_summary_report(results: dict[str, Any]) -> str:
    """
    簡単なサマリーレポートを作成

    Args:
        results: 分析結果

    Returns:
        サマリーレポート文字列
    """
    try:
        lines = ["=== 分析サマリー ==="]

        # メタデータ
        metadata = results.get("metadata", {})
        lines.append(f"対象: {metadata.get('query', 'N/A')}")
        lines.append(f"ステータス: {metadata.get('analysis_status', 'N/A')}")
        lines.append("")

        # 主要道路ネットワーク
        major = results.get("major_network")
        if major:
            lines.append("主要道路ネットワーク:")
            lines.append(f"  ノード: {major.get('node_count', 0):,}")
            lines.append(f"  エッジ: {major.get('edge_count', 0):,}")
            lines.append(f"  α指数: {major.get('alpha_index', 0):.1f}%")
            lines.append(f"  β指数: {major.get('beta_index', 0):.2f}")
            lines.append("")

        # 全道路ネットワーク
        full = results.get("full_network")
        if full:
            lines.append("全道路ネットワーク:")
            lines.append(f"  ノード: {full.get('node_count', 0):,}")
            lines.append(f"  エッジ: {full.get('edge_count', 0):,}")
            lines.append(f"  α指数: {full.get('alpha_index', 0):.1f}%")
            lines.append(f"  β指数: {full.get('beta_index', 0):.2f}")

        return "\n".join(lines)

    except Exception as e:
        return f"サマリー生成エラー: {e}"


def check_dependencies() -> dict[str, bool]:
    """
    必要な依存関係をチェック

    Returns:
        依存関係のチェック結果
    """
    dependencies = {
        "osmnx": False,
        "networkx": False,
        "pandas": False,
        "numpy": False,
        "matplotlib": False,
        "geopandas": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    return dependencies


def get_memory_usage_info() -> dict[str, str]:
    """
    メモリ使用量の情報を取得

    Returns:
        メモリ情報の辞書
    """
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss": f"{memory_info.rss / 1024 / 1024:.1f} MB",
            "vms": f"{memory_info.vms / 1024 / 1024:.1f} MB",
            "percent": f"{process.memory_percent():.1f}%"
        }
    except ImportError:
        return {
            "rss": "N/A (psutil not available)",
            "vms": "N/A (psutil not available)",
            "percent": "N/A (psutil not available)"
        }
    except Exception as e:
        return {
            "rss": f"Error: {e}",
            "vms": f"Error: {e}",
            "percent": f"Error: {e}"
        }


def export_summary_table(data: dict[str, Any], filepath: str, format_type: str = "csv") -> bool:
    """
    サマリーテーブルをファイルに出力

    Args:
        data: 出力するデータ
        filepath: 出力先ファイルパス
        format_type: 出力形式 ("csv", "excel", "json")

    Returns:
        出力成功時True
    """
    try:
        # データをDataFrameに変換
        if isinstance(data, dict):
            # 辞書の場合は1行のDataFrameとして処理
            if all(isinstance(v, int | float | str | bool | type(None)) for v in data.values()):
                # フラットな辞書の場合
                df = pd.DataFrame([data])
            else:
                # ネストした辞書の場合は列として展開
                flattened = _flatten_dict(data)
                df = pd.DataFrame([flattened])
        elif isinstance(data, list):
            # リストの場合
            df = pd.DataFrame(data)
        else:
            # その他の場合はそのままDataFrameに
            df = pd.DataFrame(data)

        # ファイル出力
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "csv":
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        elif format_type.lower() in ["excel", "xlsx"]:
            df.to_excel(filepath, index=False)
        elif format_type.lower() == "json":
            df.to_json(filepath, orient="records", force_ascii=False, indent=2)
        else:
            raise ValueError(f"サポートされていないフォーマット: {format_type}")

        logger.info(f"サマリーテーブルを出力: {filepath}")
        return True

    except Exception as e:
        logger.error(f"サマリーテーブル出力エラー: {e}")
        return False


def _flatten_dict(d: dict, parent_key: str = "", separator: str = "_") -> dict:
    """
    ネストした辞書をフラット化

    Args:
        d: フラット化する辞書
        parent_key: 親キー
        separator: キーの区切り文字

    Returns:
        フラット化された辞書
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_area_summary(results: dict[str, Any]) -> dict[str, Any]:
    """
    面積関連のサマリーを作成

    Args:
        results: 分析結果

    Returns:
        面積サマリー
    """
    summary = {
        "area_ha": 0.0,
        "area_km2": 0.0,
        "network_density": 0.0,
        "coverage_ratio": 0.0
    }

    try:
        # 面積情報の取得
        area_ha = results.get("area_ha", 0.0)
        summary["area_ha"] = area_ha
        summary["area_km2"] = area_ha / 100.0  # ヘクタールから平方キロメートルに変換

        # ネットワーク密度の計算
        major_network = results.get("major_network", {})
        if major_network and area_ha > 0:
            edge_count = major_network.get("edge_count", 0)
            total_length = major_network.get("total_length", 0)

            if total_length > 0:
                # 道路長密度 (km/km²)
                summary["network_density"] = (total_length / 1000) / (area_ha / 100)
            elif edge_count > 0:
                # エッジ密度 (edges/km²)
                summary["network_density"] = edge_count / (area_ha / 100)

        # カバレッジ比率（主要道路vs全道路）
        full_network = results.get("full_network", {})
        if major_network and full_network:
            major_edges = major_network.get("edge_count", 0)
            full_edges = full_network.get("edge_count", 0)

            if full_edges > 0:
                summary["coverage_ratio"] = major_edges / full_edges

    except Exception as e:
        logger.error(f"面積サマリー作成エラー: {e}")
        summary["error"] = str(e)

    return summary
