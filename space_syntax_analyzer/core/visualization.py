"""
可視化モジュール - NetworkVisualizer

ネットワークと分析結果の可視化機能を提供します。
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """ネットワーク可視化を行うクラス"""
    
    def __init__(self) -> None:
        """NetworkVisualizerを初期化"""
        # matplotlib日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        
    def plot_network_comparison(
        self,
        major_network: nx.Graph,
        full_network: nx.Graph | None = None,
        results: dict[str, Any] | None = None,
        save_path: str | None = None
    ) -> None:
        """
        主要道路と全道路ネットワークの比較表示
        
        Args:
            major_network: 主要道路ネットワーク
            full_network: 全道路ネットワーク
            results: 分析結果
            save_path: 保存パス
        """
        if full_network is None:
            self._plot_single_network(major_network, "主要道路ネットワーク", save_path)
            return
            
        # 2つのネットワークを並べて表示
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('道路ネットワーク比較', fontsize=16, fontweight='bold')
        
        # 主要道路ネットワーク
        self._plot_network_on_axis(
            major_network, 
            axes[0], 
            "主要道路ネットワーク (>4m)",
            'blue'
        )
        
        # 全道路ネットワーク
        self._plot_network_on_axis(
            full_network,
            axes[1], 
            "全道路ネットワーク",
            'red'
        )
        
        # 分析結果がある場合はテキストとして表示
        if results:
            self._add_metrics_text(fig, results)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def _plot_single_network(
        self,
        graph: nx.Graph,
        title: str,
        save_path: str | None = None
    ) -> None:
        """単一ネットワークの表示"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        self._plot_network_on_axis(graph, ax, title, 'blue')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def _plot_network_on_axis(
        self,
        graph: nx.Graph,
        ax: plt.Axes,
        title: str,
        color: str = 'blue'
    ) -> None:
        """指定されたaxesにネットワークを描画"""
        try:
            if graph.number_of_nodes() == 0:
                ax.text(0.5, 0.5, 'ネットワークデータなし', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
                
            # ノードとエッジの座標を取得
            pos = {}
            for node, data in graph.nodes(data=True):
                if 'x' in data and 'y' in data:
                    pos[node] = (data['x'], data['y'])
                    
            if not pos:
                ax.text(0.5, 0.5, '座標データなし', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(title)
                return
                
            # ネットワーク描画
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                edge_color=color,
                width=0.8,
                alpha=0.7
            )
            
            # 交差点（次数3以上）を強調表示
            intersections = [node for node, degree in graph.degree() if degree >= 3]
            if intersections:
                intersection_pos = {node: pos[node] for node in intersections if node in pos}
                nx.draw_networkx_nodes(
                    graph.subgraph(intersections), intersection_pos, ax=ax,
                    node_color='red',
                    node_size=30,
                    alpha=0.8
                )
                
            # 端点を表示
            endpoints = [node for node, degree in graph.degree() if degree == 1]
            if endpoints:
                endpoint_pos = {node: pos[node] for node in endpoints if node in pos}
                nx.draw_networkx_nodes(
                    graph.subgraph(endpoints), endpoint_pos, ax=ax,
                    node_color='orange',
                    node_size=20,
                    alpha=0.8
                )
                
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            
        except Exception as e:
            logger.error(f"ネットワーク描画エラー: {e}")
            ax.text(0.5, 0.5, f'描画エラー: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            
    def _add_metrics_text(self, fig: plt.Figure, results: dict[str, Any]) -> None:
        """分析結果をテキストとして図に追加"""
        text_lines = []
        
        for network_type, metrics in results.items():
            network_name = "主要道路" if network_type == "major_network" else "全道路"
            text_lines.append(f"【{network_name}】")
            text_lines.append(f"ノード数: {metrics.get('nodes', 0)}")
            text_lines.append(f"エッジ数: {metrics.get('edges', 0)}")
            text_lines.append(f"α指数: {metrics.get('alpha_index', 0):.1f}%")
            text_lines.append(f"β指数: {metrics.get('beta_index', 0):.2f}")
            text_lines.append(f"迂回率: {metrics.get('avg_circuity', 0):.2f}")
            text_lines.append("")
            
        # テキストボックスとして表示
        textstr = "\n".join(text_lines)
        fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})
        
    def plot_metrics_comparison(
        self,
        results: dict[str, Any],
        save_path: str | None = None
    ) -> None:
        """
        指標の比較チャートを作成
        
        Args:
            results: 分析結果
            save_path: 保存パス
        """
        if len(results) < 2:
            logger.warning("比較するネットワークが不足しています")
            return
            
        # 指標データの準備
        metrics_df = pd.DataFrame(results).T
        
        # レーダーチャート用のデータ準備
        radar_metrics = [
            'alpha_index', 'beta_index', 'gamma_index',
            'road_density', 'intersection_density'
        ]
        
        available_metrics = [m for m in radar_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            logger.warning("表示可能な指標がありません")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 棒グラフ
        metrics_df[available_metrics].plot(kind='bar', ax=ax1)
        ax1.set_title('指標比較（棒グラフ）', fontsize=12, fontweight='bold')
        ax1.set_ylabel('指標値')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # テーブル表示
        ax2.axis('off')
        table_data = []
        for metric in available_metrics:
            row = [metric]
            for network_type in results.keys():
                value = results[network_type].get(metric, 0)
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            table_data.append(row)
            
        headers = ['指標'] + list(results.keys())
        table = ax2.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title('指標一覧', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_network_statistics(
        self,
        graph: nx.Graph,
        title: str = "ネットワーク統計",
        save_path: str | None = None
    ) -> None:
        """
        ネットワークの基本統計を可視化
        
        Args:
            graph: ネットワークグラフ
            title: グラフタイトル
            save_path: 保存パス
        """
        if graph.number_of_nodes() == 0:
            logger.warning("空のネットワークです")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 次数分布
        degrees = [degree for node, degree in graph.degree()]
        axes[0, 0].hist(degrees, bins=max(1, len(set(degrees))), edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('次数分布')
        axes[0, 0].set_xlabel('次数')
        axes[0, 0].set_ylabel('頻度')
        
        # エッジ長分布
        edge_lengths = [data.get('length', 0) for _, _, data in graph.edges(data=True)]
        if edge_lengths:
            axes[0, 1].hist(edge_lengths, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('エッジ長分布')
            axes[0, 1].set_xlabel('長さ (m)')
            axes[0, 1].set_ylabel('頻度')
            
        # 連結成分のサイズ分布
        components = list(nx.connected_components(graph))
        component_sizes = [len(comp) for comp in components]
        axes[1, 0].bar(range(len(component_sizes)), sorted(component_sizes, reverse=True))
        axes[1, 0].set_title('連結成分サイズ')
        axes[1, 0].set_xlabel('成分番号')
        axes[1, 0].set_ylabel('ノード数')
        
        # 基本統計情報
        axes[1, 1].axis('off')
        stats_text = f"""基本統計:
ノード数: {graph.number_of_nodes()}
エッジ数: {graph.number_of_edges()}
連結成分数: {len(components)}
平均次数: {2 * graph.number_of_edges() / graph.number_of_nodes():.2f}
密度: {nx.density(graph):.4f}"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.8})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_metrics_summary_table(
        self,
        results: dict[str, Any],
        save_path: str | None = None
    ) -> pd.DataFrame:
        """
        指標のサマリーテーブルを作成
        
        Args:
            results: 分析結果
            save_path: 保存パス
            
        Returns:
            指標サマリーDataFrame
        """
        # データフレーム作成
        df = pd.DataFrame(results).T
        
        # 列名を日本語に変換
        column_mapping = {
            'nodes': 'ノード数',
            'edges': 'エッジ数', 
            'total_length_m': '道路総延長(m)',
            'area_ha': '面積(ha)',
            'mu_index': '回路指数(μ)',
            'mu_per_ha': '平均回路指数(μ/ha)',
            'alpha_index': 'α指数(%)',
            'beta_index': 'β指数',
            'gamma_index': 'γ指数(%)',
            'avg_shortest_path': '平均最短距離(m)',
            'road_density': '道路密度(m/ha)',
            'intersection_density': '交差点密度(n/ha)',
            'avg_circuity': '平均迂回率',
        }
        
        # 存在する列のみ変換
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        # 数値の丸め
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].round(2)
                
        if save_path:
            df.to_csv(save_path, encoding='utf-8-sig')
            
        return df
        
    def plot_metrics_radar_chart(
        self,
        results: dict[str, Any],
        save_path: str | None = None
    ) -> None:
        """
        指標のレーダーチャートを作成
        
        Args:
            results: 分析結果
            save_path: 保存パス
        """
        try:
            import numpy as np
            
            # レーダーチャート用の指標選択
            radar_metrics = ['alpha_index', 'beta_index', 'gamma_index']
            metric_labels = ['α指数(%)', 'β指数', 'γ指数(%)']
            
            # データの準備
            network_types = list(results.keys())
            values = []
            
            for network_type in network_types:
                network_values = []
                for metric in radar_metrics:
                    value = results[network_type].get(metric, 0)
                    # 正規化（0-1の範囲に）
                    if metric == 'alpha_index' or metric == 'gamma_index':
                        normalized_value = value / 100.0  # パーセント値
                    else:
                        normalized_value = min(value / 3.0, 1.0)  # β指数は通常3以下
                    network_values.append(normalized_value)
                values.append(network_values)
                
            # レーダーチャート作成
            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]  # 円を閉じる
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, (network_type, network_values) in enumerate(zip(network_types, values)):
                network_values += network_values[:1]  # 円を閉じる
                ax.plot(angles, network_values, 'o-', linewidth=2, 
                       label=network_type, color=colors[i % len(colors)])
                ax.fill(angles, network_values, alpha=0.25, color=colors[i % len(colors)])
                
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title('指標レーダーチャート', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            
        except ImportError:
            logger.warning("レーダーチャートの作成にはnumpyが必要です")
        except Exception as e:
            logger.error(f"レーダーチャート作成エラー: {e}")