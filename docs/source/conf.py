#!/usr/bin/env python
"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

try:
    from tomllib import load as toml_load
except ImportError:
    from tomli import load as toml_load

# project info
author = "Space Syntax Analyzer Development Team"
copyright = "2024-2025, Space Syntax Analyzer Development Team"
project = "Space Syntax Analyzer"

# go up two levels from current working dir (/docs/source) to package root
pkg_root_path = str(Path.cwd().parent.parent)
sys.path.insert(0, pkg_root_path)

# dynamically load version from pyproject.toml or fallback to default
try:
    with Path("../../pyproject.toml").open("rb") as f:
        pyproject = toml_load(f)
    version = release = pyproject["project"]["version"]
except (FileNotFoundError, KeyError):
    version = release = "0.1.0"

# mock import all required + optional dependency packages because readthedocs
# does not have them installed
autodoc_mock_imports = [
    "geopandas",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "rasterio",
    "requests",
    "rio-vrt",
    "scipy",
    "shapely",
    "sklearn",
    "osmnx",
    "space_syntax_analyzer",
]

# linkcheck for some DOI redirects gets HTTP 403 in CI environment
linkcheck_ignore = [r"https://doi\.org/.*"]

# type annotations configuration
autodoc_typehints = "description"
napoleon_use_param = True
napoleon_use_rtype = False
typehints_document_rtype = True
typehints_use_rtype = False
typehints_fully_qualified = False

# general configuration and options for HTML output
# see https://www.sphinx-doc.org/en/master/usage/configuration.html
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc", 
    "sphinx.ext.napoleon", 
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# HTML output options
html_static_path: list[str] = []

# テーマ設定 - 利用可能なテーマから選択
# 1. furo (モダンでクリーン)
# 2. alabaster (Sphinxデフォルト)
# 3. basic (最小限のテーマ)
try:
    import furo
    html_theme = "furo"
    html_theme_options = {
        "sidebar_hide_name": False,
    }
except ImportError:
    try:
        import sphinx_rtd_theme
        html_theme = "sphinx_rtd_theme"
        html_theme_options = {
            "navigation_depth": 4,
            "collapse_navigation": False,
            "sticky_navigation": True,
            "includehidden": True,
            "titles_only": False
        }
    except ImportError:
        # フォールバック to 基本テーマ
        html_theme = "alabaster"
        html_theme_options = {
            "sidebar_width": "300px",
            "page_width": "1200px",
        }

# 多言語設定
language = "ja"  # 日本語をデフォルトに
locale_dirs = ['locale/']
gettext_compact = False

# その他の設定
needs_sphinx = "7"
root_doc = "index"
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}
templates_path: list[str] = []

# 日本語検索対応
html_search_language = 'ja'

# intersphinx mapping (外部ドキュメントへのリンク)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'osmnx': ('https://osmnx.readthedocs.io/en/stable/', None),
}

# TODO 拡張の設定
todo_include_todos = True

# Autodoc 設定
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# 日本語フォント対応（必要に応じて）
# latex_elements = {
#     'fontpkg': r'''
# \usepackage{xeCJK}
# \setCJKmainfont{Noto Sans CJK JP}
# ''',
# }