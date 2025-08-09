#!/usr/bin/env python3
"""
環境診断スクリプト
"""

import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd: list) -> str:
    """コマンドを実行して結果を取得"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "エラーまたは未インストール"


def check_python_environment():
    """Python環境の診断"""
    print("🐍 Python環境:")
    print(f"  Python バージョン: {sys.version}")
    print(f"  実行パス: {sys.executable}")
    print(f"  プラットフォーム: {platform.platform()}")
    print(f"  アーキテクチャ: {platform.machine()}")


def check_uv_environment():
    """UV環境の診断"""
    print("\n📦 UV環境:")
    print(f"  uvバージョン: {run_command(['uv', '--version'])}")
    print(f"  現在のディレクトリ: {Path.cwd()}")
    
    venv_path = Path(".venv")
    print(f"  .venv存在: {venv_path.exists()}")
    
    if venv_path.exists():
        python_path = venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = venv_path / "Scripts" / "python.exe"  # Windows
        print(f"  仮想環境Python: {python_path.exists()}")


def check_system_libraries():
    """システムライブラリの診断"""
    print("\n🔧 システムライブラリ:")
    
    # macOS用
    if platform.system() == "Darwin":
        print(f"  GDAL: {run_command(['brew', 'list', 'gdal'])}")
        print(f"  PROJ: {run_command(['brew', 'list', 'proj'])}")
        print(f"  GEOS: {run_command(['brew', 'list', 'geos'])}")
    
    # Linux用
    elif platform.system() == "Linux":
        print(f"  GDAL: {run_command(['dpkg', '-l', 'libgdal-dev'])}")
        print(f"  PROJ: {run_command(['dpkg', '-l', 'libproj-dev'])}")
        print(f"  GEOS: {run_command(['dpkg', '-l', 'libgeos-dev'])}")


def check_package_installations():
    """パッケージインストール状況の診断"""
    print("\n📋 パッケージインストール状況:")
    
    packages = [
        "numpy", "scipy", "pandas", "matplotlib", 
        "shapely", "geopandas", "networkx", "osmnx"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")


def check_project_structure():
    """プロジェクト構造の診断"""
    print("\n📁 プロジェクト構造:")
    
    expected_files = [
        "pyproject.toml",
        "space_syntax_analyzer/__init__.py",
        "space_syntax_analyzer/core/__init__.py",
        "tests/test_analyzer.py"
    ]
    
    for file_path in expected_files:
        path = Path(file_path)
        print(f"  {'✅' if path.exists() else '❌'} {file_path}")


def main():
    """診断実行"""
    print("🔍 space-syntax-analyzer 環境診断")
    print("=" * 50)
    
    check_python_environment()
    check_uv_environment()
    check_system_libraries()
    check_package_installations()
    check_project_structure()
    
    print("\n💡 推奨解決手順:")
    print("1. rm -rf .venv")
    print("2. uv sync --dev")
    print("3. python check_dependencies.py")
    print("4. make test")
    
    print("\n🆘 問題が続く場合:")
    print("- macOS: brew install gdal proj geos")
    print("- Ubuntu: sudo apt-get install libgdal-dev libproj-dev libgeos-dev")
    print("- Windows: conda install -c conda-forge gdal proj geos")


if __name__ == "__main__":
    main()