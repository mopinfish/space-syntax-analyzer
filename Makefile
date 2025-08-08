# Makefile
.PHONY: install test lint format type-check clean build publish

# パッケージ管理
install:
	uv sync

install-dev:
	uv sync --dev

# テスト
test:
	uv run pytest

test-cov:
	uv run pytest --cov=space_syntax_analyzer --cov-report=html --cov-report=term

# コード品質
lint:
	uv run ruff check space_syntax_analyzer/
	uv run ruff check tests/

format:
	uv run black space_syntax_analyzer/
	uv run black tests/
	uv run isort space_syntax_analyzer/
	uv run isort tests/

type-check:
	uv run mypy space_syntax_analyzer/

# 全品質チェック
quality: format lint type-check test

# クリーンアップ
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# ビルド
build: clean
	uv build

# PyPI公開（テスト環境）
publish-test: build
	uv publish --repository testpypi

# PyPI公開（本番環境）
publish: build
	uv publish

# ドキュメント生成
docs:
	cd docs && make html

# 開発用サーバー起動
dev:
	uv run python examples/basic_usage.py
