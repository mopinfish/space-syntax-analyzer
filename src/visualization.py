"""
可視化モジュール
パス: src/visualization.py

Space Syntax解析結果の可視化機能
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString

from .config_manager import VisualizationSettings


class Visualizer:
    """可視化クラス"""
    
    def __init__(self, settings: dict):
        """
        初期化
        
        Args:
            settings: 可視化設定辞書
        """
        self.logger = logging.getLogger(__name__)
        self.settings = VisualizationSettings(**settings)
        
        # matplotlib設定
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = self.settings.figure_size
        plt.rcParams['figure.dpi'] = self.settings.dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        
        # 日本語フォント設定（利用可能な場合）
        try:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        except:
            pass
        
        self.logger.info("Visualizer初期化完了")
    
    def create_map_visualizations(self, graph: nx.Graph, metrics: Dict[str, pd.Series], 
                                 output_dir: Path):
        """
        地図可視化の作成
        
        Args:
            graph: 対象グラフ
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        self.logger.info("地図可視化作成開始")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 各指標の地図作成
            for metric_name, series in metrics.items():
                if isinstance(series, pd.Series) and len(series) > 0:
                    self._create_metric_map(
                        graph, series, metric_name, 
                        output_dir / f"map_{metric_name}"
                    )
            
            # 複合マップの作成
            self._create_composite_maps(graph, metrics, output_dir)
            
            self.logger.info("地図可視化作成完了")
            
        except Exception as e:
            self.logger.error(f"地図可視化作成エラー: {e}")
            raise
    
    def create_statistical_plots(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        統計グラフの作成
        
    def _create_statistics_table(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        統計表の作成
        
        Args:
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # Series形式の指標のみ抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series) and len(v) > 0}
            
            if not series_metrics:
                return
            
            # 統計表作成
            stats_data = []
            
            for metric_name, series in series_metrics.items():
                clean_series = series.dropna()
                if len(clean_series) > 0:
                    stats_data.append({
                        '指標': self._get_metric_label(metric_name),
                        'データ数': len(clean_series),
                        '平均': f"{clean_series.mean():.4f}",
                        '標準偏差': f"{clean_series.std():.4f}",
                        '最小値': f"{clean_series.min():.4f}",
                        '第1四分位': f"{clean_series.quantile(0.25):.4f}",
                        '中央値': f"{clean_series.median():.4f}",
                        '第3四分位': f"{clean_series.quantile(0.75):.4f}",
                        '最大値': f"{clean_series.max():.4f}",
                        '歪度': f"{clean_series.skew():.4f}",
                        '尖度': f"{clean_series.kurtosis():.4f}"
                    })
            
            stats_df = pd.DataFrame(stats_data)
            
            # CSV保存
            stats_df.to_csv(output_dir / "statistics_table.csv", 
                           index=False, encoding='utf-8-sig')
            
            # 表の可視化
            fig, ax = plt.subplots(figsize=(14, len(stats_data) * 0.8 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=stats_df.values,
                           colLabels=stats_df.columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # ヘッダーのスタイリング
            for i in range(len(stats_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 交互の行色
            for i in range(1, len(stats_data) + 1):
                for j in range(len(stats_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f2f2f2')
            
            plt.title('Space Syntax指標 統計表', fontsize=14, pad=20)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"statistics_table.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"統計表作成エラー: {e}")
            plt.close()
    
    def _get_metric_label(self, metric_name: str) -> str:
        """
        指標名の日本語ラベル取得
        
        Args:
            metric_name: 指標名
            
        Returns:
            日本語ラベル
        """
        label_map = {
            'connectivity': 'Connectivity（接続性）',
            'integration': 'Integration（統合性）',
            'choice': 'Choice（選択性）',
            'depth': 'Depth（深度）',
            'intelligibility': 'Intelligibility（理解容易性）',
            'angular_integration': 'Angular Integration（角度統合性）',
            'angular_choice': 'Angular Choice（角度選択性）',
            'angular_depth': 'Angular Depth（角度深度）'
        }
        
        # 半径付き指標の処理
        for base_name, label in label_map.items():
            if metric_name.startswith(base_name):
                if '_r' in metric_name:
                    radius = metric_name.split('_r')[-1]
                    return f"{label} (R{radius})"
                return label
        
        return metric_name
    
    def create_comparison_plots(self, results_dict: Dict[str, Dict], output_dir: Path):
        """
        複数解析結果の比較プロット作成
        
        Args:
            results_dict: 解析結果辞書（キー: 解析名, 値: 解析結果）
            output_dir: 出力ディレクトリ
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 共通指標の抽出
            common_metrics = set()
            for results in results_dict.values():
                if 'metrics' in results:
                    common_metrics.update(results['metrics'].keys())
            
            # 指標別比較
            for metric_name in common_metrics:
                self._create_metric_comparison(
                    results_dict, metric_name, output_dir
                )
            
        except Exception as e:
            self.logger.error(f"比較プロット作成エラー: {e}")
    
    def _create_metric_comparison(self, results_dict: Dict[str, Dict], 
                                 metric_name: str, output_dir: Path):
        """
        指標別比較プロットの作成
        
        Args:
            results_dict: 解析結果辞書
            metric_name: 指標名
            output_dir: 出力ディレクトリ
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            data_for_comparison = {}
            
            # データ収集
            for analysis_name, results in results_dict.items():
                if metric_name in results.get('metrics', {}):
                    series = results['metrics'][metric_name]
                    if isinstance(series, pd.Series) and len(series) > 0:
                        data_for_comparison[analysis_name] = series.dropna()
            
            if len(data_for_comparison) < 2:
                return
            
            # 1. ヒストグラム比較
            ax1 = axes[0]
            for analysis_name, data in data_for_comparison.items():
                ax1.hist(data, alpha=0.6, label=analysis_name, bins=30)
            ax1.set_title(f'{self._get_metric_label(metric_name)} - 分布比較')
            ax1.set_xlabel('値')
            ax1.set_ylabel('頻度')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 箱ひげ図比較
            ax2 = axes[1]
            box_data = [data.values for data in data_for_comparison.values()]
            box_labels = list(data_for_comparison.keys())
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_title(f'{self._get_metric_label(metric_name)} - 箱ひげ図比較')
            ax2.set_ylabel('値')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # 3. 統計比較
            ax3 = axes[2]
            stats_comparison = []
            for analysis_name, data in data_for_comparison.items():
                stats_comparison.append({
                    '解析': analysis_name,
                    '平均': data.mean(),
                    '標準偏差': data.std(),
                    '中央値': data.median()
                })
            
            stats_df = pd.DataFrame(stats_comparison)
            x_pos = np.arange(len(stats_df))
            
            width = 0.25
            ax3.bar(x_pos - width, stats_df['平均'], width, label='平均', alpha=0.8)
            ax3.bar(x_pos, stats_df['標準偏差'], width, label='標準偏差', alpha=0.8)
            ax3.bar(x_pos + width, stats_df['中央値'], width, label='中央値', alpha=0.8)
            
            ax3.set_title(f'{self._get_metric_label(metric_name)} - 統計値比較')
            ax3.set_xlabel('解析タイプ')
            ax3.set_ylabel('値')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(stats_df['解析'], rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 散布図（最初の2つの解析）
            ax4 = axes[3]
            if len(data_for_comparison) >= 2:
                analysis_names = list(data_for_comparison.keys())
                data1 = data_for_comparison[analysis_names[0]]
                data2 = data_for_comparison[analysis_names[1]]
                
                # 共通インデックスでマージ
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) > 0:
                    x_data = data1.loc[common_idx]
                    y_data = data2.loc[common_idx]
                    
                    ax4.scatter(x_data, y_data, alpha=0.6, s=20)
                    
                    # 回帰直線
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax4.plot(x_data, p(x_data), "r--", alpha=0.8)
                    
                    # 相関係数
                    corr = x_data.corr(y_data)
                    ax4.text(0.05, 0.95, f'相関係数: {corr:.3f}', 
                           transform=ax4.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax4.set_xlabel(f'{analysis_names[0]}')
                    ax4.set_ylabel(f'{analysis_names[1]}')
                    ax4.set_title(f'{self._get_metric_label(metric_name)} - 散布図')
                    ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'{self._get_metric_label(metric_name)} 比較分析', fontsize=14)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"comparison_{metric_name}.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"指標比較プロット作成エラー ({metric_name}): {e}")
            plt.close()
    
    def create_interactive_map(self, graph: nx.Graph, metrics: Dict[str, pd.Series], 
                              output_path: Path):
        """
        インタラクティブマップの作成（Folium使用）
        
        Args:
            graph: グラフ
            metrics: 指標辞書
            output_path: 出力HTMLパス
        """
        try:
            import folium
            from folium import plugins
            
            # グラフから座標範囲取得
            coords = []
            for node, data in graph.nodes(data=True):
                if 'geometry' in data:
                    geom = data['geometry']
                    if hasattr(geom, 'coords'):
                        coords.extend(list(geom.coords))
                    elif hasattr(geom, 'x'):
                        coords.append((geom.x, geom.y))
            
            if not coords:
                return
            
            # 中心座標計算
            x_coords, y_coords = zip(*coords)
            center_x, center_y = np.mean(x_coords), np.mean(y_coords)
            
            # 座標変換（UTMから緯度経度へ）
            # 注意: 実際の実装では適切な座標変換が必要
            center_lat, center_lon = center_y / 111000, center_x / 111000  # 簡易変換
            
            # Foliumマップ作成
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles='OpenStreetMap'
            )
            
            # 指標選択用の層グループ作成
            for metric_name, series in metrics.items():
                if not isinstance(series, pd.Series) or len(series) == 0:
                    continue
                
                feature_group = folium.FeatureGroup(name=self._get_metric_label(metric_name))
                
                # 色の正規化
                values = series.dropna()
                if len(values) == 0:
                    continue
                
                vmin, vmax = values.min(), values.max()
                if vmin == vmax:
                    continue
                
                # 線分の描画
                for node, data in graph.nodes(data=True):
                    if 'geometry' not in data:
                        continue
                    
                    geometry = data['geometry']
                    metric_value = series.get(node, np.nan)
                    
                    if pd.isna(metric_value):
                        continue
                    
                    # 正規化された値で色決定
                    norm_value = (metric_value - vmin) / (vmax - vmin)
                    color = plt.cm.viridis(norm_value)
                    color_hex = mcolors.to_hex(color)
                    
                    if isinstance(geometry, LineString):
                        # 座標変換
                        coords_geo = [(y/111000, x/111000) for x, y in geometry.coords]
                        
                        folium.PolyLine(
                            coords_geo,
                            color=color_hex,
                            weight=3,
                            opacity=0.8,
                            popup=f'{self._get_metric_label(metric_name)}: {metric_value:.4f}'
                        ).add_to(feature_group)
                
                feature_group.add_to(m)
            
            # レイヤーコントロール追加
            folium.LayerControl().add_to(m)
            
            # HTML保存
            m.save(str(output_path))
            
            self.logger.info(f"インタラクティブマップ作成完了: {output_path}")
            
        except ImportError:
            self.logger.warning("Foliumがインストールされていません。インタラクティブマップをスキップします。")
        except Exception as e:
            self.logger.error(f"インタラクティブマップ作成エラー: {e}")
_dir: 出力ディレクトリ
        """
        self.logger.info("統計グラフ作成開始")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ヒストグラム
            self._create_histograms(metrics, output_dir)
            
            # 散布図行列
            self._create_scatter_matrix(metrics, output_dir)
            
            # 相関ヒートマップ
            self._create_correlation_heatmap(metrics, output_dir)
            
            # 箱ひげ図
            self._create_boxplots(metrics, output_dir)
            
            # 統計サマリー表
            self._create_statistics_table(metrics, output_dir)
            
            self.logger.info("統計グラフ作成完了")
            
        except Exception as e:
            self.logger.error(f"統計グラフ作成エラー: {e}")
            raise
    
    def _create_metric_map(self, graph: nx.Graph, metric_series: pd.Series, 
                          metric_name: str, output_path: Path):
        """
        指標別地図の作成
        
        Args:
            graph: グラフ
            metric_series: 指標データ
            metric_name: 指標名
            output_path: 出力パス（拡張子なし）
        """
        try:
            fig, ax = plt.subplots(figsize=self.settings.figure_size)
            
            # 色の正規化
            values = metric_series.dropna()
            if len(values) == 0:
                self.logger.warning(f"指標 {metric_name} にデータがありません")
                return
            
            vmin, vmax = values.min(), values.max()
            if vmin == vmax:
                vmin, vmax = vmin - 0.1, vmax + 0.1
            
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.get_cmap(self.settings.color_map)
            
            # ノード（線分）の描画
            for node, data in graph.nodes(data=True):
                if 'geometry' not in data:
                    continue
                
                geometry = data['geometry']
                metric_value = metric_series.get(node, np.nan)
                
                if pd.isna(metric_value):
                    color = 'gray'
                    alpha = 0.3
                else:
                    color = cmap(norm(metric_value))
                    alpha = self.settings.alpha
                
                if isinstance(geometry, LineString):
                    x_coords, y_coords = geometry.xy
                    ax.plot(x_coords, y_coords, 
                           color=color, 
                           linewidth=self.settings.edge_linewidth,
                           alpha=alpha)
                elif isinstance(geometry, Point):
                    ax.scatter(geometry.x, geometry.y,
                             c=[color], 
                             s=self.settings.node_size,
                             alpha=alpha)
            
            # カラーバー
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(self._get_metric_label(metric_name), rotation=270, labelpad=20)
            
            # 軸設定
            ax.set_aspect('equal')
            ax.set_title(f'{self._get_metric_label(metric_name)} 分布図', fontsize=14, pad=20)
            ax.set_xlabel('X座標 (m)')
            ax.set_ylabel('Y座標 (m)')
            
            # 軸の調整
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(f"{output_path}.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
            self.logger.debug(f"指標マップ作成完了: {metric_name}")
            
        except Exception as e:
            self.logger.error(f"指標マップ作成エラー ({metric_name}): {e}")
            plt.close()
    
    def _create_composite_maps(self, graph: nx.Graph, metrics: Dict[str, pd.Series], 
                              output_dir: Path):
        """
        複合マップの作成
        
        Args:
            graph: グラフ
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # 主要指標の複合表示
            main_metrics = ['connectivity', 'integration', 'choice']
            available_metrics = [m for m in main_metrics if m in metrics]
            
            if len(available_metrics) < 2:
                return
            
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric_name in enumerate(available_metrics):
                ax = axes[i]
                metric_series = metrics[metric_name]
                
                # 色の正規化
                values = metric_series.dropna()
                if len(values) == 0:
                    continue
                
                vmin, vmax = values.min(), values.max()
                if vmin == vmax:
                    vmin, vmax = vmin - 0.1, vmax + 0.1
                
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.get_cmap(self.settings.color_map)
                
                # 描画
                for node, data in graph.nodes(data=True):
                    if 'geometry' not in data:
                        continue
                    
                    geometry = data['geometry']
                    metric_value = metric_series.get(node, np.nan)
                    
                    if pd.isna(metric_value):
                        color = 'gray'
                        alpha = 0.3
                    else:
                        color = cmap(norm(metric_value))
                        alpha = self.settings.alpha
                    
                    if isinstance(geometry, LineString):
                        x_coords, y_coords = geometry.xy
                        ax.plot(x_coords, y_coords, 
                               color=color, 
                               linewidth=self.settings.edge_linewidth,
                               alpha=alpha)
                
                ax.set_aspect('equal')
                ax.set_title(self._get_metric_label(metric_name))
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # カラーバー
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, shrink=0.8)
            
            plt.suptitle('Space Syntax指標 比較', fontsize=16)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"composite_map.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"複合マップ作成エラー: {e}")
            plt.close()
    
    def _create_histograms(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        ヒストグラムの作成
        
        Args:
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # Series形式の指標のみ抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series) and len(v) > 0}
            
            if not series_metrics:
                return
            
            n_metrics = len(series_metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for i, (metric_name, series) in enumerate(series_metrics.items()):
                ax = axes[i]
                
                # ヒストグラム描画
                series.dropna().hist(ax=ax, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # 統計線
                mean_val = series.mean()
                median_val = series.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', label=f'平均: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'中央値: {median_val:.3f}')
                
                ax.set_title(self._get_metric_label(metric_name))
                ax.set_xlabel('値')
                ax.set_ylabel('頻度')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 不要な軸を非表示
            for i in range(len(series_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Space Syntax指標 分布', fontsize=16)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"histograms.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"ヒストグラム作成エラー: {e}")
            plt.close()
    
    def _create_scatter_matrix(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        散布図行列の作成
        
        Args:
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # Series形式の指標のみ抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series) and len(v) > 0}
            
            if len(series_metrics) < 2:
                return
            
            # DataFrameに変換
            df = pd.DataFrame(series_metrics)
            df = df.dropna()
            
            if len(df) == 0:
                return
            
            # 散布図行列作成
            n_vars = len(df.columns)
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
            
            for i, var1 in enumerate(df.columns):
                for j, var2 in enumerate(df.columns):
                    ax = axes[i, j]
                    
                    if i == j:
                        # 対角線: ヒストグラム
                        df[var1].hist(ax=ax, bins=20, alpha=0.7, color='lightblue')
                        ax.set_title(self._get_metric_label(var1), fontsize=10)
                    else:
                        # 散布図
                        ax.scatter(df[var2], df[var1], alpha=0.6, s=10)
                        
                        # 回帰直線
                        z = np.polyfit(df[var2], df[var1], 1)
                        p = np.poly1d(z)
                        ax.plot(df[var2], p(df[var2]), "r--", alpha=0.8)
                        
                        # 相関係数
                        corr = df[var1].corr(df[var2])
                        ax.text(0.05, 0.95, f'r={corr:.3f}', 
                               transform=ax.transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    if i == n_vars - 1:
                        ax.set_xlabel(self._get_metric_label(var2), fontsize=10)
                    if j == 0:
                        ax.set_ylabel(self._get_metric_label(var1), fontsize=10)
                    
                    ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.suptitle('Space Syntax指標 散布図行列', fontsize=14)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"scatter_matrix.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"散布図行列作成エラー: {e}")
            plt.close()
    
    def _create_correlation_heatmap(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        相関ヒートマップの作成
        
        Args:
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # Series形式の指標のみ抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series) and len(v) > 0}
            
            if len(series_metrics) < 2:
                return
            
            # DataFrameに変換
            df = pd.DataFrame(series_metrics)
            correlation_matrix = df.corr()
            
            # ヒートマップ作成
            plt.figure(figsize=(10, 8))
            
            # 日本語ラベル作成
            japanese_labels = [self._get_metric_label(col) for col in correlation_matrix.columns]
            
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       xticklabels=japanese_labels,
                       yticklabels=japanese_labels,
                       cbar_kws={'label': '相関係数'})
            
            plt.title('Space Syntax指標 相関行列', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"correlation_heatmap.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"相関ヒートマップ作成エラー: {e}")
            plt.close()
    
    def _create_boxplots(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        箱ひげ図の作成
        
        Args:
            metrics: 指標辞書
            output_dir: 出力ディレクトリ
        """
        try:
            # Series形式の指標のみ抽出
            series_metrics = {k: v for k, v in metrics.items() 
                             if isinstance(v, pd.Series) and len(v) > 0}
            
            if not series_metrics:
                return
            
            # データ準備
            data_for_boxplot = []
            labels = []
            
            for metric_name, series in series_metrics.items():
                data_for_boxplot.append(series.dropna().values)
                labels.append(self._get_metric_label(metric_name))
            
            # 箱ひげ図作成
            plt.figure(figsize=(12, 6))
            
            bp = plt.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
            
            # カラーリング
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_boxplot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('Space Syntax指標 箱ひげ図', fontsize=14)
            plt.ylabel('値')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存
            for ext in self.settings.save_formats:
                plt.savefig(output_dir / f"boxplots.{ext}", 
                           dpi=self.settings.dpi, 
                           bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"箱ひげ図作成エラー: {e}")
            plt.close()
    
    def _create_statistics_table(self, metrics: Dict[str, pd.Series], output_dir: Path):
        """
        統計表の作成
        
        Args:
            metrics: 指標辞書
            output