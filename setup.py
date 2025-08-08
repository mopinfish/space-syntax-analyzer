#!/usr/bin/env python3
"""
Space Syntax解析システム セットアップスクリプト
パス: setup.py

パッケージのインストールと配布用設定
"""

from setuptools import setup, find_packages
from pathlib import Path

# プロジェクトルートディレクトリ
here = Path(__file__).parent.resolve()

# README.mdの内容を読み込み
long_description = (here / "README.md").read_text(encoding="utf-8")

# requirements.txtから依存関係を読み込み
def parse_requirements(filename):
    """requirements.txtファイルを解析して依存関係リストを生成"""
    requirements = []
    with open(here / filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # コメント行と空行をスキップ
            if line and not line.startswith('#'):
                # インライン コメントを削除
                if '#' in line:
                    line = line.split('#')[0].strip()
                if line:
                    requirements.append(line)
    return requirements

# 基本依存関係
install_requires = [
    "networkx>=2.5",
    "osmnx>=1.0.0",
    "geopandas>=0.10.0", 
    "pandas>=1.4.0",
    "numpy>=1.22.0",
    "scipy>=1.8.0",
    "shapely>=2.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "jinja2>=3.0.0",
    "requests>=2.27.0,<3.0.0"
]

# オプション依存関係
extras_require = {
    'full': [
        "folium>=0.12.0",
        "weasyprint>=54.0",
        "psutil>=5.8.0",
        "plotly>=5.0.0",
        "rasterio>=1.3.0",
        "contextily>=1.2.0",
        "tqdm>=4.60.0",
        "rich>=12.0.0"
    ],
    'pdf': [
        "weasyprint>=54.0"
    ],
    'interactive': [
        "folium>=0.12.0",
        "plotly>=5.0.0"
    ],
    'dev': [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0", 
        "flake8>=4.0.0",
        "mypy>=0.950",
        "jupyter>=1.0.0",
        "memory_profiler>=0.60.0"
    ],
    'docs': [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.17.0"
    ]
}

# フルインストール用（開発者向け）
extras_require['all'] = list(set(
    extras_require['full'] + 
    extras_require['dev'] + 
    extras_require['docs']
))

setup(
    # プロジェクト基本情報
    name="space-syntax-analyzer",
    version="1.0.0",
    description="OpenStreetMapデータを活用したSpace Syntax理論に基づく都市空間構造解析システム",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # 開発者情報
    author="Space Syntax Development Team",
    author_email="dev@example.com",
    
    # プロジェクトURL
    url="https://github.com/your-org/space-syntax-analyzer",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/space-syntax-analyzer/issues",
        "Documentation": "https://space-syntax-analyzer.readthedocs.io/",
        "Source Code": "https://github.com/your-org/space-syntax-analyzer",
    },
    
    # パッケージ設定
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
        "config": ["*.json"],
    },
    include_package_data=True,
    
    # Python バージョン要件
    python_requires=">=3.8",
    
    # 依存関係
    install_requires=install_requires,
    extras_require=extras_require,
    
    # コンソールスクリプト
    entry_points={
        "console_scripts": [
            "space-syntax=main:main",
            "ss-analyze=main:main",
        ],
    },
    
    # データファイル
    data_files=[
        ("config", ["config/default_config.json"]),
    ],
    
    # PyPI 分類子
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education", 
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: Japanese",
        "Natural Language :: English",
    ],
    
    # キーワード
    keywords=[
        "space syntax", "urban analysis", "network analysis", 
        "openstreetmap", "gis", "spatial analysis", "graph theory",
        "urban planning", "architecture", "accessibility",
        "integration", "connectivity", "choice", "intelligibility"
    ],
    
    # ライセンス
    license="MIT",
    
    # 開発ステータス
    zip_safe=False,
    
    # テスト設定
    test_suite="tests",
    tests_require=extras_require["dev"],
    
    # オプション設定
    options={
        "bdist_wheel": {
            "universal": False,  # Python 3専用
        },
    },
)

# インストール後の設定チェック
def post_install_check():
    """インストール後の動作確認"""
    try:
        print("🔍 依存関係チェック中...")
        
        import networkx
        import osmnx  
        import geopandas
        import pandas
        import numpy
        import matplotlib
        
        print("✅ 基本依存関係: OK")
        
        # オプション依存関係チェック
        optional_packages = {
            'folium': 'インタラクティブマップ機能',
            'weasyprint': 'PDF生成機能', 
            'psutil': 'システム監視機能',
            'plotly': '高度な可視化機能'
        }
        
        for package, description in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package}: OK ({description})")
            except ImportError:
                print(f"⚠️  {package}: 未インストール ({description})")
        
        print("\n🎉 Space Syntax解析システムのセットアップが完了しました！")
        print("\n📝 使用例:")
        print("   python main.py --place 'Tokyo, Japan'")
        print("   space-syntax --place 'Kyoto, Japan' --analysis-type both")
        print("\n📚 詳細なドキュメントは docs/ ディレクトリを参照してください。")
        
    except ImportError as e:
        print(f"❌ セットアップエラー: {e}")
        print("pip install -r requirements.txt を実行してください。")

if __name__ == "__main__":
    # 直接実行時は依存関係チェックのみ
    post_install_check()
