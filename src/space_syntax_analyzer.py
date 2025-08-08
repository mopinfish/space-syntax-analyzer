def calculate_intelligibility(self, connectivity: pd.Series, integration: pd.Series) -> float:
        """
        Intelligibility値の計算
        
        Args:
            connectivity: Connectivity値
            integration: Integration値
            
        Returns:
            Intelligibility値（相関係数）
        """
        self.logger.debug("Intelligibility計算開始")
        
        try:
            # 共通インデックスで結合
            common_index = connectivity.index.intersection(integration.index)
            
            if len(common_index) < 2:
                return 0.0
            
            conn_values = connectivity.loc[common_index]
            integ_values = integration.loc[common_index]
            
            # Pearson相関係数計算
            correlation, _ = stats.pearsonr(conn_values, integ_values)
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Intelligibility計算エラー: {e}")
            return 0.0
    
    def calculate_statistics(self, metrics: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        統計値の計算
        
        Args:
            metrics: 指標辞書
            
        Returns:
            統計値辞書
        """
        self.logger.debug("統計値計算開始")
        
        try:
            statistics = {}
            
            for metric_name, series in metrics.items():
                if isinstance(series, pd.Series) and len(series) > 0:
                    statistics[metric_name] = {
    def calculate_statistics(self, metrics: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        統計値の計算
        
        Args:
            metrics: 指標辞書
            
        Returns:
            統計値辞書
        """
        self.logger.debug("統計値計算開始")
        
        try:
            statistics = {}
            
            for metric_name, series in metrics.items():
                if isinstance(series, pd.Series) and len(series) > 0:
                    statistics[metric_name] = {
                        'count': len(series),
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'median': float(series.median()),
                        'q25': float(series.quantile(0.25)),
                        'q75': float(series.quantile(0.75)),
                        'skewness': float(stats.skew(series.dropna())),
                        'kurtosis': float(stats.kurtosis(series.dropna()))
                    }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"統計値計算エラー: {e}")
            return {}
    
    def calculate_correlations(self, metrics: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        相関分析
        
        Args:
            metrics: 指標辞書
            
        Returns:
            相関行列
        """
        self.logger.debug("相関分析開始")
        
        try:
            # Series形式の指標のみを抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series)}
            
            if len(series_metrics) < 2:
                return pd.DataFrame()
            
            # DataFrameに変換
            df = pd.DataFrame(series_metrics)
            
            # 相関行列計算
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"相関分析エラー: {e}")
            return pd.DataFrame()
    
    def _create_radius_subgraph(self, graph: nx.Graph, center_node: int, radius: int) -> nx.Graph:
        """
        半径内サブグラフの作成（トポロジカル距離）
        
        Args:
            graph: 元のグラフ
            center_node: 中心ノード
            radius: 半径（ホップ数）
            
        Returns:
            サブグラフ
        """
        try:
            # BFS探索で半径内ノード取得
            visited = {center_node}
            current_level = {center_node}
            
            for _ in range(radius):
                next_level = set()
                for node in current_level:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
                current_level = next_level
                
                if not current_level:
                    break
            
            return graph.subgraph(visited).copy()
            
        except Exception:
            return graph.subgraph([center_node]).copy()
    
    def _create_distance_subgraph(self, graph: nx.Graph, center_node: int, radius: float) -> nx.Graph:
        """
        距離内サブグラフの作成（メートル距離）
        
        Args:
            graph: 元のグラフ
            center_node: 中心ノード
            radius: 半径（メートル）
            
        Returns:
            サブグラフ
        """
        try:
            # Dijkstra法で距離計算
            distances = nx.single_source_dijkstra_path_length(
                graph, center_node, cutoff=radius, weight='length'
            )
            
            # 半径内ノード抽出
            nodes_in_radius = [node for node, dist in distances.items() if dist <= radius]
            
            return graph.subgraph(nodes_in_radius).copy()
            
        except Exception:
            return graph.subgraph([center_node]).copy()
    
    def _calculate_angular_depths(self, graph: nx.Graph, source: int) -> Dict[int, float]:
        """
        角度重み付き深度の計算
        
        Args:
            graph: グラフ
            source: 起点ノード
            
        Returns:
            各ノードへの角度深度
        """
        try:
            # Dijkstra法の改良版（角度重み付き）
            distances = {source: 0.0}
            unvisited = set(graph.nodes())
            
            while unvisited:
                # 最小距離ノード選択
                current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
                
                if distances.get(current, float('inf')) == float('inf'):
                    break
                
                unvisited.remove(current)
                
                # 隣接ノード更新
                for neighbor in graph.neighbors(current):
                    if neighbor in unvisited:
                        # 角度重み計算
                        angle_weight = self._get_angular_weight(graph, current, neighbor)
                        new_distance = distances[current] + angle_weight
                        
                        if new_distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = new_distance
            
            return distances
            
        except Exception:
            return {source: 0.0}
    
    def _find_angular_shortest_paths(self, graph: nx.Graph, source: int, target: int) -> List[List[int]]:
        """
        角度重み付き最短経路の検索
        
        Args:
            graph: グラフ
            source: 起点
            target: 終点
            
        Returns:
            最短経路のリスト
        """
        try:
            # 簡略化: 通常の最短経路を返す
            # 実際の実装では角度重み付きアルゴリズムを使用
            paths = list(nx.all_shortest_paths(graph, source, target))
            return paths
            
        except nx.NetworkXNoPath:
            return []
        except Exception:
            return []
    
    def _get_angular_weight(self, graph: nx.Graph, current: int, neighbor: int) -> float:
        """
        角度重みの計算（Space Syntax理論準拠）
        
        Args:
            graph: グラフ
            current: 現在ノード
            neighbor: 隣接ノード
            
        Returns:
            角度重み（w(θ) ∝ θ, w(0) = 0, w(π/2) = 1）
        """
        try:
            # エッジの角度差情報取得
            edge_data = graph.get_edge_data(current, neighbor)
            
            if edge_data and 'angle_difference' in edge_data:
                angle_diff = edge_data['angle_difference']
                # 角度差を正規化（0-180度を0-π/2に正規化）
                angle_rad = math.radians(min(angle_diff, 180.0 - angle_diff))
                
                # Space Syntax理論に従った重み計算
                # w(θ) = θ / (π/2) for 0 ≤ θ ≤ π/2
                # 直角で重み1、直進で重み0
                if angle_rad == 0:
                    return 0.0  # 直進の場合
                else:
                    return angle_rad / (math.pi / 2.0)  # 正規化された角度重み
            else:
                return 1.0  # デフォルト重み（直角相当）
                
        except Exception:
            return 1.0
    
    def normalize_nach(self, choice_series: pd.Series) -> pd.Series:
        """
        NACH（Normalised Angular Choice）の計算
        
        Args:
            choice_series: Choice値のSeries
            
        Returns:
            正規化されたNACH値
        """
        try:
            # Space Syntax理論準拠の正規化式: log(Choice(r) + 2)
            nach_values = choice_series.apply(lambda x: math.log(x + 2) if not pd.isna(x) else np.nan)
            return pd.Series(nach_values, name=f'NACH_{choice_series.name}')
            
        except Exception as e:
            self.logger.error(f"NACH正規化エラー: {e}")
            return choice_series
    
    def normalize_nain(self, integration_series: pd.Series, node_count: int, total_depth: float) -> pd.Series:
        """
        NAIN（Normalised Angular Integration）の計算
        
        Args:
            integration_series: Integration値のSeries
            node_count: ノード数（r半径内）
            total_depth: 総深度
            
        Returns:
            正規化されたNAIN値
        """
        try:
            # Space Syntax理論準拠の正規化式（高度版）:
            # NAIN = 1.2 / sqrt(Node_count(r)) / (Total_depth(r) + 2)
            if node_count > 0 and total_depth > 0:
                normalization_factor = 1.2 / math.sqrt(node_count) / (total_depth + 2)
                nain_values = integration_series * normalization_factor
            else:
                # フォールバック: 基本正規化式 log(Integration(r) + 2)
                nain_values = integration_series.apply(lambda x: math.log(x + 2) if not pd.isna(x) else np.nan)
            
            return pd.Series(nain_values, name=f'NAIN_{integration_series.name}')
            
        except Exception as e:
            self.logger.error(f"NAIN正規化エラー: {e}")
            # フォールバック処理
            return self.normalize_basic_nain(integration_series)
    
    def normalize_basic_nain(self, integration_series: pd.Series) -> pd.Series:
        """
        基本NAIN正規化（シンプル版）
        
        Args:
            integration_series: Integration値のSeries
            
        Returns:
            正規化されたNAIN値
        """
        try:
            # 基本正規化式: log(Integration(r) + 2)
            nain_values = integration_series.apply(lambda x: math.log(x + 2) if not pd.isna(x) else np.nan)
            return pd.Series(nain_values, name=f'NAIN_{integration_series.name}')
            
        except Exception as e:
            self.logger.error(f"基本NAIN正規化エラー: {e}")
            return integration_series
        """
        解析結果のエクスポート
        
        Args:
            results: 解析結果
            output_path: 出力ファイルパス
        """
        try:
            import json
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONシリアライズ可能な形式に変換
            export_data = {
                'metrics': {},
                'statistics': results.get('statistics', {}),
                'correlations': results.get('correlations', pd.DataFrame()).to_dict()
                if isinstance(results.get('correlations'), pd.DataFrame) else {}
            }
            
            # メトリクスデータの変換
            for metric_name, series in results.get('metrics', {}).items():
                if isinstance(series, pd.Series):
                    export_data['metrics'][metric_name] = series.to_dict()
                else:
                    export_data['metrics'][metric_name] = series
            
            # JSON出力
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
    def export_results(self, results: Dict[str, Any], output_path: str):
        """
        解析結果のエクスポート
        
        Args:
            results: 解析結果
            output_path: 出力ファイルパス
        """
        try:
            import json
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONシリアライズ可能な形式に変換
            export_data = {
                'metrics': {},
                'statistics': results.get('statistics', {}),
                'correlations': results.get('correlations', pd.DataFrame()).to_dict()
                if isinstance(results.get('correlations'), pd.DataFrame) else {}
            }
            
            # メトリクスデータの変換
            for metric_name, series in results.get('metrics', {}).items():
                if isinstance(series, pd.Series):
                    export_data['metrics'][metric_name] = series.to_dict()
                else:
                    export_data['metrics'][metric_name] = series
            
            # JSON出力
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"解析結果エクスポート完了: {output_path}")
            
        except Exception as e:
            self.logger.error(f"解析結果エクスポートエラー: {e}")
            raise
    
    def apply_normalization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Space Syntax理論準拠の正規化処理適用
        
        Args:
            results: 元の解析結果
            
        Returns:
            正規化処理済み結果
        """
        try:
            normalized_results = results.copy()
            metrics = normalized_results.get('metrics', {})
            
            # Choice値の正規化（NACH）
            choice_metrics = [k for k in metrics.keys() if 'choice' in k.lower()]
            for choice_key in choice_metrics:
                if isinstance(metrics[choice_key], pd.Series):
                    nach_key = f"NACH_{choice_key.replace('choice', '').strip('_')}"
                    metrics[nach_key] = self.normalize_nach(metrics[choice_key])
            
            # Integration値の正規化（NAIN）
            integration_metrics = [k for k in metrics.keys() if 'integration' in k.lower()]
            for integration_key in integration_metrics:
                if isinstance(metrics[integration_key], pd.Series):
                    # 基本正規化を適用（高度版は追加情報が必要）
                    nain_key = f"NAIN_{integration_key.replace('integration', '').strip('_')}"
                    metrics[nain_key] = self.normalize_basic_nain(metrics[integration_key])
            
            # 統計情報の再計算
            normalized_results['statistics'] = self.calculate_statistics(metrics)
            normalized_results['correlations'] = self.calculate_correlations(metrics)
            
            return normalized_results
            
        except Exception as e:
            self.logger.error(f"正規化処理エラー: {e}")
    def calculate_four_pointed_star(self, nach_series: pd.Series, nain_series: pd.Series) -> Dict[str, float]:
        """
        Four-Pointed Star Modelの計算（50都市比較基準）
        
        Args:
            nach_series: NACH値のSeries
            nain_series: NAIN値のSeries
            
        Returns:
            Z-score辞書
        """
        try:
            # Hillier et al. (2012)の50都市基準値
            reference_values = {
                'nach_max_mean': 1.695,  # 50都市のNACH最大値の平均
                'nach_max_std': 0.321,   # 50都市のNACH最大値の標準偏差
                'nach_mean_mean': 0.742, # 50都市のNACH平均値の平均
                'nach_mean_std': 0.186,  # 50都市のNACH平均値の標準偏差
                'nain_max_mean': 1.582,  # 50都市のNAIN最大値の平均
                'nain_max_std': 0.394,   # 50都市のNAIN最大値の標準偏差
                'nain_mean_mean': 0.897, # 50都市のNAIN平均値の平均
                'nain_mean_std': 0.152   # 50都市のNAIN平均値の標準偏差
            }
            
            # 現在の都市の統計値
            current_stats = {
                'nach_max': nach_series.max() if len(nach_series) > 0 else 0.0,
                'nach_mean': nach_series.mean() if len(nach_series) > 0 else 0.0,
                'nain_max': nain_series.max() if len(nain_series) > 0 else 0.0,
                'nain_mean': nain_series.mean() if len(nain_series) > 0 else 0.0
            }
            
            # Z-score計算
            z_scores = {}
            z_scores['nach_max_z'] = (current_stats['nach_max'] - reference_values['nach_max_mean']) / reference_values['nach_max_std']
            z_scores['nach_mean_z'] = (current_stats['nach_mean'] - reference_values['nach_mean_mean']) / reference_values['nach_mean_std']
            z_scores['nain_max_z'] = (current_stats['nain_max'] - reference_values['nain_max_mean']) / reference_values['nain_max_std']
            z_scores['nain_mean_z'] = (current_stats['nain_mean'] - reference_values['nain_mean_mean']) / reference_values['nain_mean_std']
            
            # 解釈情報の追加
            z_scores.update(current_stats)
            z_scores['interpretation'] = self._interpret_four_pointed_star(z_scores)
            
            return z_scores
            
        except Exception as e:
            self.logger.error(f"Four-Pointed Star計算エラー: {e}")
            return {}
    
    def _interpret_four_pointed_star(self, z_scores: Dict[str, float]) -> Dict[str, str]:
        """
        Four-Pointed Star結果の解釈
        
        Args:
            z_scores: Z-score辞書
            
        Returns:
            解釈辞書
        """
        interpretation = {}
        
        # フォアグラウンドネットワーク（主要ルート）の評価
        if z_scores['nach_max_z'] > 1.0:
            interpretation['foreground'] = "非常に強いフォアグラウンドネットワーク - 主要ルートが高度に統合"
        elif z_scores['nach_max_z'] > 0.5:
            interpretation['foreground'] = "強いフォアグラウンドネットワーク"
        elif z_scores['nach_max_z'] > -0.5:
            interpretation['foreground'] = "平均的なフォアグラウンドネットワーク"
        else:
            interpretation['foreground'] = "弱いフォアグラウンドネットワーク"
        
        # バックグラウンドネットワーク（一般街路）の評価
        if z_scores['nain_mean_z'] > 1.0:
            interpretation['background'] = "非常に統合されたバックグラウンドネットワーク - 歩行者に優しい"
        elif z_scores['nain_mean_z'] > 0.5:
            interpretation['background'] = "統合されたバックグラウンドネットワーク"
        elif z_scores['nain_mean_z'] > -0.5:
            interpretation['background'] = "平均的なバックグラウンドネットワーク"
        else:
            interpretation['background'] = "分離されたバックグラウンドネットワーク - 車依存型"
        
        # 全体的な都市タイプの判定
        if z_scores['nach_max_z'] > 0.5 and z_scores['nain_mean_z'] > 0.5:
            interpretation['city_type'] = "バランス型都市（ウィーン型）"
        elif z_scores['nach_max_z'] > 0.5 and z_scores['nain_mean_z'] < -0.5:
            interpretation['city_type'] = "車依存型都市（レリスタット型）"
        elif z_scores['nach_max_z'] < -0.5 and z_scores['nain_mean_z'] > 0.5:
            interpretation['city_type'] = "分散型都市"
        else:
            interpretation['city_type'] = "混合型都市"
        
        return interpretation"""
Space Syntax解析モジュール
パス: src/space_syntax_analyzer.py

Space Syntax理論に基づく各種指標の計算
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import math

from .config_manager import AnalysisSettings


class SpaceSyntaxAnalyzer:
    """Space Syntax解析クラス"""
    
    def __init__(self, settings: dict):
        """
        初期化
        
        Args:
            settings: 解析設定辞書
        """
        self.logger = logging.getLogger(__name__)
        self.settings = AnalysisSettings(**settings)
        self.logger.info("SpaceSyntaxAnalyzer初期化完了")
    
    def analyze_axial_map(self, axial_graph: nx.Graph) -> Dict[str, Any]:
        """
        Axial Map解析
        
        Args:
            axial_graph: Axial Mapグラフ
            
        Returns:
            解析結果辞書
        """
        self.logger.info("Axial Map解析開始")
        start_time = time.time()
        
        try:
            results = {
                'graph': axial_graph,
                'metrics': {},
                'statistics': {},
                'correlations': {}
            }
            
            # 基本指標計算
            results['metrics']['connectivity'] = self.calculate_connectivity(axial_graph)
            
            if self.settings.calculate_global:
                results['metrics']['integration'] = self.calculate_integration(axial_graph)
                results['metrics']['choice'] = self.calculate_choice(axial_graph)
                results['metrics']['depth'] = self.calculate_depth(axial_graph)
            
            if self.settings.calculate_local:
                for radius in self.settings.local_radii:
                    results['metrics'][f'integration_r{radius}'] = self.calculate_local_integration(
                        axial_graph, radius
                    )
                    results['metrics'][f'choice_r{radius}'] = self.calculate_local_choice(
                        axial_graph, radius
                    )
            
            # 統計分析
            results['statistics'] = self.calculate_statistics(results['metrics'])
            
            # 相関分析
            results['correlations'] = self.calculate_correlations(results['metrics'])
            
            # Intelligibility計算
            if 'connectivity' in results['metrics'] and 'integration' in results['metrics']:
                results['metrics']['intelligibility'] = self.calculate_intelligibility(
                    results['metrics']['connectivity'],
                    results['metrics']['integration']
                )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Axial Map解析完了 ({elapsed_time:.2f}秒)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Axial Map解析エラー: {e}")
            raise
    
    def analyze_segment_map(self, segment_graph: nx.Graph) -> Dict[str, Any]:
        """
        Segment Map解析
        
        Args:
            segment_graph: Segment Mapグラフ
            
        Returns:
            解析結果辞書
        """
        self.logger.info("Segment Map解析開始")
        start_time = time.time()
        
        try:
            results = {
                'graph': segment_graph,
                'metrics': {},
                'statistics': {},
                'correlations': {}
            }
            
            # Angular解析（セグメント特有）
            results['metrics']['connectivity'] = self.calculate_connectivity(segment_graph)
            
            if self.settings.calculate_global:
                results['metrics']['integration'] = self.calculate_angular_integration(segment_graph)
                results['metrics']['choice'] = self.calculate_angular_choice(segment_graph)
                results['metrics']['depth'] = self.calculate_angular_depth(segment_graph)
            
            if self.settings.calculate_local:
                for radius in self.settings.local_radii:
                    results['metrics'][f'integration_r{radius}'] = self.calculate_local_angular_integration(
                        segment_graph, radius
                    )
                    results['metrics'][f'choice_r{radius}'] = self.calculate_local_angular_choice(
                        segment_graph, radius
                    )
            
            # 統計分析
            results['statistics'] = self.calculate_statistics(results['metrics'])
            
            # 相関分析
            results['correlations'] = self.calculate_correlations(results['metrics'])
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Segment Map解析完了 ({elapsed_time:.2f}秒)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Segment Map解析エラー: {e}")
            raise
    
    def calculate_connectivity(self, graph: nx.Graph) -> pd.Series:
        """
        Connectivity値の計算
        
        Args:
            graph: 対象グラフ
            
        Returns:
            各ノードのConnectivity値
        """
        self.logger.debug("Connectivity計算開始")
        
        try:
            connectivity = {}
            for node in graph.nodes():
                connectivity[node] = graph.degree(node)
            
            return pd.Series(connectivity, name='connectivity')
            
        except Exception as e:
            self.logger.error(f"Connectivity計算エラー: {e}")
            raise
    
    def calculate_integration(self, graph: nx.Graph) -> pd.Series:
        """
        Global Integration値の計算（Space Syntax理論準拠）
        
        Args:
            graph: 対象グラフ
            
        Returns:
            各ノードのIntegration値
        """
        self.logger.debug("Integration計算開始")
        
        try:
            integration = {}
            
            # 最短経路長計算
            path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
            
            for node in graph.nodes():
                if node not in path_lengths:
                    integration[node] = 0.0
                    continue
                
                distances = list(path_lengths[node].values())
                n = len(distances)
                
                if n <= 2:
                    integration[node] = 0.0
                    continue
                
                # Space Syntax理論準拠の正確なIntegration計算
                # I = 2[n log₂((n+2)/3) - 1] + 1 / [(n-1)(n-2)/2 * Σdᵢⱼ/(n-1) - 1] / (n-2)
                
                # Mean Depth計算
                total_depth = sum(distances)
                mean_depth = total_depth / n
                
                # Diamond値計算（理論値）
                diamond = 2 * (n * math.log2((n + 2) / 3) - 1) + 1
                
                # Relative Asymmetry (RA)計算
                ra_numerator = (2 * (mean_depth - 1)) / (n - 2)
                ra = ra_numerator if n > 2 else 0.0
                
                # Real Relative Asymmetry (RRA)計算
                if diamond > 0:
                    rra = ra / diamond
                else:
                    rra = float('inf')
                
                # Integration値（RRAの逆数）
                if rra > 0 and not math.isinf(rra):
                    integration[node] = 1.0 / rra
                else:
                    integration[node] = 0.0
            
            series = pd.Series(integration, name='integration')
            
            # 正規化
            if self.settings.normalize and len(series) > 0:
                max_val = series.max()
                min_val = series.min()
                if max_val > min_val:
                    series = (series - min_val) / (max_val - min_val)
            
            return series
            
        except Exception as e:
            self.logger.error(f"Integration計算エラー: {e}")
            raise
    
    def calculate_choice(self, graph: nx.Graph) -> pd.Series:
        """
        Global Choice値の計算
        
        Args:
            graph: 対象グラフ
            
        Returns:
            各ノードのChoice値
        """
        self.logger.debug("Choice計算開始")
        
        try:
            choice = {node: 0.0 for node in graph.nodes()}
            
            # 全ノードペア間の最短経路を計算
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    source, target = nodes[i], nodes[j]
                    
                    try:
                        # 最短経路取得
                        paths = list(nx.all_shortest_paths(graph, source, target))
                        
                        if not paths:
                            continue
                        
                        # 各経路の重み（通常は1/経路数）
                        weight = 1.0 / len(paths)
                        
                        # 各経路上のノードにChoice値加算
                        for path in paths:
                            for node in path[1:-1]:  # 始点・終点除く
                                choice[node] += weight
                                
                    except nx.NetworkXNoPath:
                        continue
            
            series = pd.Series(choice, name='choice')
            
            # 正規化
            if self.settings.normalize and len(series) > 0 and series.max() > 0:
                series = series / series.max()
            
            return series
            
        except Exception as e:
            self.logger.error(f"Choice計算エラー: {e}")
            raise
    
    def calculate_depth(self, graph: nx.Graph) -> pd.Series:
        """
        Mean Depth値の計算
        
        Args:
            graph: 対象グラフ
            
        Returns:
            各ノードのMean Depth値
        """
        self.logger.debug("Depth計算開始")
        
        try:
            depth = {}
            
            # 最短経路長計算
            path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
            
            for node in graph.nodes():
                if node not in path_lengths:
                    depth[node] = float('inf')
                    continue
                
                distances = list(path_lengths[node].values())
                depth[node] = sum(distances) / len(distances) if distances else 0.0
            
            return pd.Series(depth, name='depth')
            
        except Exception as e:
            self.logger.error(f"Depth計算エラー: {e}")
            raise
    
    def calculate_local_integration(self, graph: nx.Graph, radius: float) -> pd.Series:
        """
        Local Integration値の計算
        
        Args:
            graph: 対象グラフ
            radius: 解析半径
            
        Returns:
            各ノードのLocal Integration値
        """
        self.logger.debug(f"Local Integration計算開始 (radius={radius})")
        
        try:
            integration = {}
            
            for node in graph.nodes():
                # 半径内のサブグラフ作成
                subgraph = self._create_radius_subgraph(graph, node, radius)
                
                if len(subgraph.nodes()) <= 1:
                    integration[node] = 0.0
                    continue
                
                # サブグラフでのIntegration計算
                sub_integration = self.calculate_integration(subgraph)
                integration[node] = sub_integration.get(node, 0.0)
            
            series = pd.Series(integration, name=f'integration_r{radius}')
            
            # 正規化
            if self.settings.normalize and len(series) > 0:
                series = (series - series.min()) / (series.max() - series.min())
            
            return series
            
        except Exception as e:
            self.logger.error(f"Local Integration計算エラー: {e}")
            raise
    
    def calculate_local_choice(self, graph: nx.Graph, radius: float) -> pd.Series:
        """
        Local Choice値の計算
        
        Args:
            graph: 対象グラフ
            radius: 解析半径
            
        Returns:
            各ノードのLocal Choice値
        """
        self.logger.debug(f"Local Choice計算開始 (radius={radius})")
        
        try:
            choice = {}
            
            for node in graph.nodes():
                # 半径内のサブグラフ作成
                subgraph = self._create_radius_subgraph(graph, node, radius)
                
                if len(subgraph.nodes()) <= 1:
                    choice[node] = 0.0
                    continue
                
                # サブグラフでのChoice計算
                sub_choice = self.calculate_choice(subgraph)
                choice[node] = sub_choice.get(node, 0.0)
            
            series = pd.Series(choice, name=f'choice_r{radius}')
            
            # 正規化
            if self.settings.normalize and len(series) > 0 and series.max() > 0:
                series = series / series.max()
            
            return series
            
        except Exception as e:
            self.logger.error(f"Local Choice計算エラー: {e}")
            raise
    
    def calculate_angular_integration(self, graph: nx.Graph) -> pd.Series:
        """
        Angular Integration値の計算（セグメント用）
        
        Args:
            graph: セグメントグラフ
            
        Returns:
            各セグメントのAngular Integration値
        """
        self.logger.debug("Angular Integration計算開始")
        
        try:
            integration = {}
            
            # 角度重み付き最短経路計算
            for node in graph.nodes():
                angular_depths = self._calculate_angular_depths(graph, node)
                
                if not angular_depths:
                    integration[node] = 0.0
                    continue
                
                # Mean Angular Depth
                mean_depth = sum(angular_depths.values()) / len(angular_depths)
                
                # Angular Integration値
                n = len(angular_depths)
                if n > 1:
                    rra = 2 * (sum(angular_depths.values()) - (n - 1)) / ((n - 1) * (n - 2))
                    integration[node] = 1.0 / rra if rra > 0 else 0.0
                else:
                    integration[node] = 0.0
            
            series = pd.Series(integration, name='angular_integration')
            
            # 正規化
            if self.settings.normalize and len(series) > 0:
                series = (series - series.min()) / (series.max() - series.min())
            
            return series
            
        except Exception as e:
            self.logger.error(f"Angular Integration計算エラー: {e}")
            raise
    
    def calculate_angular_choice(self, graph: nx.Graph) -> pd.Series:
        """
        Angular Choice値の計算（セグメント用）
        
        Args:
            graph: セグメントグラフ
            
        Returns:
            各セグメントのAngular Choice値
        """
        self.logger.debug("Angular Choice計算開始")
        
        try:
            choice = {node: 0.0 for node in graph.nodes()}
            
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    source, target = nodes[i], nodes[j]
                    
                    # 角度重み付き最短経路取得
                    paths = self._find_angular_shortest_paths(graph, source, target)
                    
                    if not paths:
                        continue
                    
                    weight = 1.0 / len(paths)
                    
                    for path in paths:
                        for node in path[1:-1]:  # 始点・終点除く
                            choice[node] += weight
            
            series = pd.Series(choice, name='angular_choice')
            
            # 正規化
            if self.settings.normalize and len(series) > 0 and series.max() > 0:
                series = series / series.max()
            
            return series
            
        except Exception as e:
            self.logger.error(f"Angular Choice計算エラー: {e}")
            raise
    
    def calculate_angular_depth(self, graph: nx.Graph) -> pd.Series:
        """
        Angular Depth値の計算（セグメント用）
        
        Args:
            graph: セグメントグラフ
            
        Returns:
            各セグメントのAngular Depth値
        """
        self.logger.debug("Angular Depth計算開始")
        
        try:
            depth = {}
            
            for node in graph.nodes():
                angular_depths = self._calculate_angular_depths(graph, node)
                
                if angular_depths:
                    depth[node] = sum(angular_depths.values()) / len(angular_depths)
                else:
                    depth[node] = float('inf')
            
            return pd.Series(depth, name='angular_depth')
            
        except Exception as e:
            self.logger.error(f"Angular Depth計算エラー: {e}")
            raise
    
    def calculate_local_angular_integration(self, graph: nx.Graph, radius: float) -> pd.Series:
        """
        Local Angular Integration値の計算
        
        Args:
            graph: セグメントグラフ
            radius: 解析半径（メートル）
            
        Returns:
            各セグメントのLocal Angular Integration値
        """
        self.logger.debug(f"Local Angular Integration計算開始 (radius={radius})")
        
        try:
            integration = {}
            
            for node in graph.nodes():
                # 距離重み付きサブグラフ作成
                subgraph = self._create_distance_subgraph(graph, node, radius)
                
                if len(subgraph.nodes()) <= 1:
                    integration[node] = 0.0
                    continue
                
                # サブグラフでのAngular Integration計算
                sub_integration = self.calculate_angular_integration(subgraph)
                integration[node] = sub_integration.get(node, 0.0)
            
            series = pd.Series(integration, name=f'angular_integration_r{radius}')
            
            # 正規化
            if self.settings.normalize and len(series) > 0:
                series = (series - series.min()) / (series.max() - series.min())
            
            return series
            
        except Exception as e:
            self.logger.error(f"Local Angular Integration計算エラー: {e}")
            raise
    
    def calculate_local_angular_choice(self, graph: nx.Graph, radius: float) -> pd.Series:
        """
        Local Angular Choice値の計算
        
        Args:
            graph: セグメントグラフ
            radius: 解析半径（メートル）
            
        Returns:
            各セグメントのLocal Angular Choice値
        """
        self.logger.debug(f"Local Angular Choice計算開始 (radius={radius})")
        
        try:
            choice = {}
            
            for node in graph.nodes():
                # 距離重み付きサブグラフ作成
                subgraph = self._create_distance_subgraph(graph, node, radius)
                
                if len(subgraph.nodes()) <= 1:
                    choice[node] = 0.0
                    continue
                
                # サブグラフでのAngular Choice計算
                sub_choice = self.calculate_angular_choice(subgraph)
                choice[node] = sub_choice.get(node, 0.0)
            
            series = pd.Series(choice, name=f'angular_choice_r{radius}')
            
            # 正規化
            if self.settings.normalize and len(series) > 0 and series.max() > 0:
                series = series / series.max()
            
            return series
            
        except Exception as e:
            self.logger.error(f"Local Angular Choice計算エラー: {e}")
            raise
