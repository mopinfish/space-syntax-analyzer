#!/usr/bin/env python3
"""
クイック修正スクリプト
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """コマンドを実行"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ エラー: {e}")
        print(f"   出力: {e.stdout}")
        print(f"   エラー: {e.stderr}")
        return False


def main():
    """修正実行"""
    print("🚀 space-syntax-analyzer クイック修正")
    print("=" * 50)
    
    # 現在のディレクトリ確認
    current_dir = Path.cwd()
    print(f"📁 作業ディレクトリ: {current_dir}")
    
    # pyproject.tomlの存在確認
    if not (current_dir / "pyproject.toml").exists():
        print("❌ pyproject.tomlが見つかりません")
        return
    
    # 1. uvの依存関係同期
    if not run_command(["uv", "sync", "--dev"], "依存関係の同期"):
        print("⚠️ uv sync が失敗しました。手動でインストールを試行します...")
        
        # 手動で重要なパッケージをインストール
        packages = ["shapely", "geopandas", "osmnx", "ruff"]
        for package in packages:
            run_command(["uv", "add", package], f"{package}のインストール")
    
    # 2. パッケージの確認
    print("\n📦 インストール済みパッケージ確認:")
    result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        installed_packages = result.stdout
        required_packages = ["shapely", "geopandas", "osmnx", "networkx", "pandas", "numpy"]
        
        for package in required_packages:
            if package in installed_packages.lower():
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package}")
    
    # 3. Pythonパスの設定を含むテスト実行
    print("\n🧪 テスト実行:")
    
    # テストファイルをuv runで実行
    test_cmd = ["uv", "run", "python", "tests/test_analyzer_lightweight.py"]
    if run_command(test_cmd, "軽量テスト実行"):
        print("✅ 基本テスト成功")
    else:
        print("❌ テストが失敗しました")
    
    # 4. 次のステップを提案
    print(f"\n🎯 次のステップ:")
    print("1. uv run make test        # 完全なテスト実行")
    print("2. uv run make lint        # コード品質チェック")
    print("3. uv run make quality     # 全体品質チェック")
    print("\nまたは:")
    print("uv run python check_dependencies.py  # 依存関係再確認")


if __name__ == "__main__":
    main()
