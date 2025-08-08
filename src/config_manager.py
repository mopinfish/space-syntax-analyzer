"""
設定管理モジュール
パス: src/config_manager.py

システム全体の設定を管理するクラス
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class OSMSettings:
    """OpenStreetMapデータ取得設定"""
    network_type: str = 'walk'  # 'walk', 'drive', 'bike', 'all'
    simplify: bool = True
    retain_all: bool = False
    truncate_by_edge: bool = True
    timeout: int = 180
    memory: Optional[int] = None
    max_query_area_size: int = 50000000


@dataclass
class GraphSettings:
    """グラフ構築設定"""
    min_segment_length: float = 10.0  # 最小セグメント長（メートル）
    angle_threshold: float = 30.0     # 角度閾値（度）
    connection_tolerance: float = 5.0  # 接続許容距離（メートル）
    simplify_tolerance: float = 1.0   # 簡略化許容距離（メートル）


@dataclass
class AnalysisSettings:
    """Space Syntax解析設定"""
    radius_values: list = None  # 解析半径のリスト
    calculate_global: bool = True
    calculate_local: bool = True
    local_radii: list = None
    weight_attribute: str = 'length'
    normalize: bool = True
    
    def __post_init__(self):
        if self.radius_values is None:
            self.radius_values = [3, 5, 7, 10]
        if self.local_radii is None:
            self.local_radii = [400, 800, 1200]


@dataclass
class VisualizationSettings:
    """可視化設定"""
    figure_size: tuple = (12, 8)
    dpi: int = 300
    color_map: str = 'viridis'
    edge_linewidth: float = 0.5
    node_size: float = 1.0
    alpha: float = 0.8
    save_formats: list = None
    
    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['png', 'svg']


@dataclass
class ReportSettings:
    """レポート生成設定"""
    include_maps: bool = True
    include_statistics: bool = True
    include_correlations: bool = True
    language: str = 'ja'  # 'ja' or 'en'
    template_path: Optional[str] = None
    logo_path: Optional[str] = None


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_path: str = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self._config = None
        
        # デフォルト設定
        self._default_config = {
            'osm_settings': asdict(OSMSettings()),
            'graph_settings': asdict(GraphSettings()),
            'analysis_settings': asdict(AnalysisSettings()),
            'visualization_settings': asdict(VisualizationSettings()),
            'report_settings': asdict(ReportSettings())
        }
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイルの読み込み"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # デフォルト設定をユーザー設定で更新
                self._config = self._deep_update(
                    self._default_config.copy(), 
                    user_config
                )
                
                self.logger.info(f"設定ファイル読み込み完了: {self.config_path}")
                
            except Exception as e:
                self.logger.warning(
                    f"設定ファイル読み込みエラー: {e}. デフォルト設定を使用します。"
                )
                self._config = self._default_config.copy()
        else:
            self.logger.info("デフォルト設定を使用します")
            self._config = self._default_config.copy()
        
        self._validate_config()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """辞書の深い更新"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _validate_config(self):
        """設定値の検証"""
        try:
            # OSM設定の検証
            osm_settings = self._config['osm_settings']
            if osm_settings['network_type'] not in ['walk', 'drive', 'bike', 'all']:
                raise ValueError(f"Invalid network_type: {osm_settings['network_type']}")
            
            if osm_settings['timeout'] <= 0:
                raise ValueError(f"timeout must be positive: {osm_settings['timeout']}")
            
            # グラフ設定の検証
            graph_settings = self._config['graph_settings']
            if graph_settings['min_segment_length'] <= 0:
                raise ValueError(f"min_segment_length must be positive: {graph_settings['min_segment_length']}")
            
            if not (0 <= graph_settings['angle_threshold'] <= 180):
                raise ValueError(f"angle_threshold must be between 0 and 180: {graph_settings['angle_threshold']}")
            
            # 解析設定の検証
            analysis_settings = self._config['analysis_settings']
            if not analysis_settings['radius_values']:
                raise ValueError("radius_values cannot be empty")
            
            if not analysis_settings['local_radii']:
                raise ValueError("local_radii cannot be empty")
            
            # 可視化設定の検証
            viz_settings = self._config['visualization_settings']
            if viz_settings['dpi'] <= 0:
                raise ValueError(f"dpi must be positive: {viz_settings['dpi']}")
            
            if not (0 <= viz_settings['alpha'] <= 1):
                raise ValueError(f"alpha must be between 0 and 1: {viz_settings['alpha']}")
            
            self.logger.info("設定値の検証完了")
            
        except Exception as e:
            self.logger.error(f"設定値検証エラー: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """設定の取得"""
        return self._config.copy()
    
    def get_osm_settings(self) -> OSMSettings:
        """OSM設定の取得"""
        return OSMSettings(**self._config['osm_settings'])
    
    def get_graph_settings(self) -> GraphSettings:
        """グラフ設定の取得"""
        return GraphSettings(**self._config['graph_settings'])
    
    def get_analysis_settings(self) -> AnalysisSettings:
        """解析設定の取得"""
        return AnalysisSettings(**self._config['analysis_settings'])
    
    def get_visualization_settings(self) -> VisualizationSettings:
        """可視化設定の取得"""
        return VisualizationSettings(**self._config['visualization_settings'])
    
    def get_report_settings(self) -> ReportSettings:
        """レポート設定の取得"""
        return ReportSettings(**self._config['report_settings'])
    
    def update_setting(self, section: str, key: str, value: Any):
        """設定値の更新"""
        if section not in self._config:
            raise ValueError(f"Unknown section: {section}")
        
        self._config[section][key] = value
        self.logger.info(f"設定更新: {section}.{key} = {value}")
    
    def save_config(self, output_path: str = None):
        """設定ファイルの保存"""
        save_path = Path(output_path) if output_path else self.config_path
        
        if not save_path:
            raise ValueError("保存パスが指定されていません")
        
        try:
            # 出力ディレクトリ作成
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定ファイル保存完了: {save_path}")
            
        except Exception as e:
            self.logger.error(f"設定ファイル保存エラー: {e}")
            raise
    
    def create_default_config_file(self, output_path: str):
        """デフォルト設定ファイルの作成"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._default_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"デフォルト設定ファイル作成完了: {output_path}")
            
        except Exception as e:
            self.logger.error(f"デフォルト設定ファイル作成エラー: {e}")
            raise