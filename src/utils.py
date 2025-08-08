"""
ユーティリティモジュール
パス: src/utils.py

共通機能とヘルパー関数
"""

import logging
import sys
import os
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional
import json
import pickle
import gzip
import time
from datetime import datetime
import numpy as np
import pandas as pd


def setup_logging(level: str = 'INFO', quiet: bool = False, 
                 log_file: Optional[str] = None) -> None:
    """
    ログ設定のセットアップ
    
    Args:
        level: ログレベル
        quiet: 静音モード
        log_file: ログファイルパス
    """
    # ログレベル設定
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # ログフォーマット
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ルートロガー設定
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソールハンドラー
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # ファイルハンドラー
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 外部ライブラリのログレベル調整
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    座標の有効性検証
    
    Args:
        lat: 緯度
        lon: 経度
        
    Returns:
        有効性
    """
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)


def calculate_distance(coord1: Tuple[float, float], 
                      coord2: Tuple[float, float]) -> float:
    """
    2点間の距離計算（ハーヴァサイン公式）
    
    Args:
        coord1: 座標1 (lat, lon)
        coord2: 座標2 (lat, lon)
        
    Returns:
        距離（メートル）
    """
    from math import radians, sin, cos, sqrt, atan2
    
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # 地球の半径（メートル）
    R = 6371000
    
    return R * c


def format_file_size(size_bytes: int) -> str:
    """
    ファイルサイズの人間読みやすい形式への変換
    
    Args:
        size_bytes: バイト数
        
    Returns:
        フォーマット済みサイズ
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    実行時間の人間読みやすい形式への変換
    
    Args:
        seconds: 秒数
        
    Returns:
        フォーマット済み時間
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def save_json(data: Any, file_path: Union[str, Path], 
              indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    JSONファイルの保存
    
    Args:
        data: 保存するデータ
        file_path: ファイルパス
        indent: インデント
        ensure_ascii: ASCII強制
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                 default=json_serializer)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    JSONファイルの読み込み
    
    Args:
        file_path: ファイルパス
        
    Returns:
        読み込んだデータ
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザー（numpy型対応）
    
    Args:
        obj: シリアライズ対象オブジェクト
        
    Returns:
        シリアライズ可能な形式
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_pickle(data: Any, file_path: Union[str, Path], 
                compress: bool = True) -> None:
    """
    Pickleファイルの保存
    
    Args:
        data: 保存するデータ
        file_path: ファイルパス
        compress: 圧縮の有無
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: Union[str, Path], 
                compressed: bool = None) -> Any:
    """
    Pickleファイルの読み込み
    
    Args:
        file_path: ファイルパス
        compressed: 圧縮ファイルかどうか（自動判定）
        
    Returns:
        読み込んだデータ
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    # 圧縮の自動判定
    if compressed is None:
        compressed = file_path.suffix.lower() in ['.gz', '.gzip']
    
    try:
        if compressed:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        # 圧縮形式の自動切替
        try:
            if not compressed:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except:
            raise e


def create_progress_bar(total: int, description: str = "Processing") -> 'ProgressBar':
    """
    プログレスバーの作成
    
    Args:
        total: 総数
        description: 説明
        
    Returns:
        プログレスバーオブジェクト
    """
    return ProgressBar(total, description)


class ProgressBar:
    """シンプルなプログレスバークラス"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, amount: int = 1) -> None:
        """
        プログレス更新
        
        Args:
            amount: 更新量
        """
        self.current += amount
        current_time = time.time()
        
        # 更新頻度制御（0.1秒間隔）
        if current_time - self.last_update >= 0.1 or self.current >= self.total:
            self._display()
            self.last_update = current_time
    
    def _display(self) -> None:
        """プログレス表示"""
        if self.total <= 0:
            return
        
        percent = min(100, (self.current / self.total) * 100)
        bar_length = 50
        filled_length = int(bar_length * self.current // self.total)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed_time * (self.total - self.current) / self.current
            eta_str = format_duration(eta)
        else:
            eta_str = "未定"
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% '
              f'({self.current}/{self.total}) ETA: {eta_str}', end='', flush=True)
        
        if self.current >= self.total:
            print()  # 改行
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current < self.total:
            self.current = self.total
            self._display()


class Timer:
    """実行時間計測クラス"""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """計測開始"""
        self.start_time = time.time()
        if self.description:
            print(f"{self.description} 開始...")
    
    def stop(self) -> float:
        """
        計測終了
        
        Returns:
            経過時間（秒）
        """
        self.end_time = time.time()
        elapsed = self.elapsed_time()
        
        if self.description:
            print(f"{self.description} 完了 ({format_duration(elapsed)})")
        
        return elapsed
    
    def elapsed_time(self) -> float:
        """
        経過時間取得
        
        Returns:
            経過時間（秒）
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def memory_usage() -> Dict[str, float]:
    """
    メモリ使用量の取得
    
    Returns:
        メモリ使用量情報
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss': 0, 'vms': 0, 'percent': 0}


def check_dependencies() -> Dict[str, bool]:
    """
    依存関係のチェック
    
    Returns:
        依存関係の状況
    """
    dependencies = {
        'networkx': False,
        'osmnx': False,
        'geopandas': False,
        'pandas': False,
        'numpy': False,
        'matplotlib': False,
        'scipy': False,
        'shapely': False,
        'folium': False,
        'seaborn': False,
        'psutil': False,
        'weasyprint': False,
        'pdfkit': False
    }
    
    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies


def print_system_info() -> None:
    """システム情報の表示"""
    import platform
    
    print("="*50)
    print("システム情報")
    print("="*50)
    print(f"Python バージョン: {platform.python_version()}")
    print(f"プラットフォーム: {platform.platform()}")
    print(f"アーキテクチャ: {platform.architecture()[0]}")
    print(f"プロセッサ: {platform.processor()}")
    
    # メモリ情報
    memory = memory_usage()
    if memory['rss'] > 0:
        print(f"メモリ使用量: {memory['rss']:.1f} MB ({memory['percent']:.1f}%)")
    
    print("\n依存関係チェック:")
    print("-"*30)
    
    deps = check_dependencies()
    for package, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{status} {package}")
    
    print("="*50)


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = False,
                      create_parent: bool = False) -> Path:
    """
    ファイルパスの検証
    
    Args:
        file_path: ファイルパス
        must_exist: 存在必須
        create_parent: 親ディレクトリ作成
        
    Returns:
        検証済みパス
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"ファイルが存在しません: {path}")
    
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    安全な除算（0除算対応）
    
    Args:
        numerator: 分子
        denominator: 分母
        default: デフォルト値
        
    Returns:
        除算結果
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def clean_dataframe(df: pd.DataFrame, 
                   drop_na: bool = True,
                   drop_duplicates: bool = True,
                   reset_index: bool = True) -> pd.DataFrame:
    """
    DataFrameのクリーニング
    
    Args:
        df: 対象DataFrame
        drop_na: 欠損値削除
        drop_duplicates: 重複削除
        reset_index: インデックスリセット
        
    Returns:
        クリーニング済みDataFrame
    """
    result = df.copy()
    
    if drop_na:
        result = result.dropna()
    
    if drop_duplicates:
        result = result.drop_duplicates()
    
    if reset_index:
        result = result.reset_index(drop=True)
    
    return result


def normalize_series(series: pd.Series, 
                    method: str = 'minmax') -> pd.Series:
    """
    Series の正規化
    
    Args:
        series: 対象Series
        method: 正規化方法 ('minmax', 'zscore', 'robust')
        
    Returns:
        正規化済みSeries
    """
    if method == 'minmax':
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    
    elif method == 'robust':
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(0, index=series.index)
        return (series - median) / mad
    
    else:
        raise ValueError(f"未知の正規化方法: {method}")


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    ファイル情報の取得
    
    Args:
        file_path: ファイルパス
        
    Returns:
        ファイル情報
    """
    path = Path(file_path)
    
    if not path.exists():
        return {'exists': False}
    
    stat = path.stat()
    
    return {
        'exists': True,
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'suffix': path.suffix,
        'name': path.name,
        'stem': path.stem
    }


def backup_file(file_path: Union[str, Path], 
               backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    ファイルのバックアップ
    
    Args:
        file_path: バックアップ対象ファイル
        backup_dir: バックアップディレクトリ
        
    Returns:
        バックアップファイルパス
    """
    import shutil
    
    source_path = Path(file_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"バックアップ対象ファイルが存在しません: {source_path}")
    
    if backup_dir is None:
        backup_dir = source_path.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # タイムスタンプ付きバックアップファイル名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
    backup_path = backup_dir / backup_name
    
    shutil.copy2(source_path, backup_path)
    return backup_path


class ConfigValidator:
    """設定値バリデータ"""
    
    @staticmethod
    def validate_positive_number(value: Union[int, float], 
                                name: str = "value") -> None:
        """正の数値の検証"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} は正の数値である必要があります: {value}")
    
    @staticmethod
    def validate_range(value: Union[int, float], 
                      min_val: Union[int, float], 
                      max_val: Union[int, float],
                      name: str = "value") -> None:
        """範囲の検証"""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} は {min_val} から {max_val} の範囲である必要があります: {value}")
    
    @staticmethod
    def validate_choice(value: Any, 
                       choices: List[Any],
                       name: str = "value") -> None:
        """選択肢の検証"""
        if value not in choices:
            raise ValueError(f"{name} は {choices} のいずれかである必要があります: {value}")
    
    @staticmethod
    def validate_file_extension(file_path: Union[str, Path],
                               allowed_extensions: List[str],
                               name: str = "file") -> None:
        """ファイル拡張子の検証"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in [e.lower() for e in allowed_extensions]:
            raise ValueError(f"{name} の拡張子は {allowed_extensions} のいずれかである必要があります: {ext}")


# エラーハンドリング用デコレータ
def handle_errors(logger: Optional[logging.Logger] = None):
    """
    エラーハンドリングデコレータ
    
    Args:
        logger: ロガー
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"{func.__name__} でエラーが発生しました: {e}")
                raise
        return wrapper
    return decorator


# パフォーマンス測定デコレータ
def measure_performance(logger: Optional[logging.Logger] = None):
    """
    パフォーマンス測定デコレータ
    
    Args:
        logger: ロガー
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if logger:
                logger.info(f"{func.__name__} 実行時間: {format_duration(elapsed_time)}")
            
            return result
        return wrapper
    return decorator