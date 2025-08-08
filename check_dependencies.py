#!/usr/bin/env python3
"""
依存関係のチェックとインストール確認スクリプト
"""

import importlib
import subprocess
import sys
from typing import Dict, List


def check_dependency(module_name: str, import_name: str = None) -> bool:
    """依存関係をチェック"""
    try:
        if import_name:
            importlib.import_module(import_name)
        else:
            importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def install_missing_packages(missing_packages: List[str]) -> None:
    """不足しているパッケージをインストール"""
    if not missing_packages:
        return
        
    print(f"\n🔧 不足パッケージのインストール: {', '.join(missing_packages)}")
    
    try:
        # uvを使用してインストール
        subprocess.run(
            ['uv', 'add'] + missing_packages,
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ インストール完了")
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        print("手動でインストールしてください:")
        print(f"uv add {' '.join(missing_packages)}")
    except FileNotFoundError:
        print("❌ uvが見つかりません。pipでインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")


def main():
    """メイン処理"""
    print("🔍 space-syntax-analyzer 依存関係チェック")
    print("=" * 50)
    
    # 必要な依存関係の定義
    dependencies: Dict[str, str] = {
        'networkx': 'networkx',
        'numpy': 'numpy', 
        'scipy': 'scipy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib.pyplot',
        'shapely': 'shapely.geometry',
        'geopandas': 'geopandas',
        'osmnx': 'osmnx',
    }
    
    # 開発用依存関係
    dev_dependencies: Dict[str, str] = {
        'pytest': 'pytest',
        'black': 'black',
        'isort': 'isort',
        'mypy': 'mypy',
        'ruff': 'ruff',
    }
    
    missing_packages = []
    missing_dev_packages = []
    
    print("📦 必須依存関係チェック:")
    for package, import_name in dependencies.items():
        if check_dependency(package, import_name):
            print(f"  ✅ {package}")
        else:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    print("\n🛠️  開発用依存関係チェック:")
    for package, import_name in dev_dependencies.items():
        if check_dependency(package, import_name):
            print(f"  ✅ {package}")
        else:
            print(f"  ❌ {package}")
            missing_dev_packages.append(package)
    
    # 結果サマリー
    print(f"\n📊 チェック結果:")
    print(f"  必須パッケージ: {len(dependencies) - len(missing_packages)}/{len(dependencies)} ✅")
    print(f"  開発パッケージ: {len(dev_dependencies) - len(missing_dev_packages)}/{len(dev_dependencies)} ✅")
    
    # インストール提案
    if missing_packages:
        print(f"\n⚠️  不足している必須パッケージ: {', '.join(missing_packages)}")
        install_missing_packages(missing_packages)
    
    if missing_dev_packages:
        print(f"\n⚠️  不足している開発パッケージ: {', '.join(missing_dev_packages)}")
        print("開発用パッケージをインストール:")
        print("uv sync --dev")
    
    # テスト実行提案
    if not missing_packages:
        print("\n🎯 次のステップ:")
        print("1. make test     # テスト実行")
        print("2. make lint     # コード品質チェック") 
        print("3. make format   # コードフォーマット")
    else:
        print("\n🔄 依存関係インストール後に再実行してください")


if __name__ == "__main__":
    main()