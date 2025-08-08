.PHONY: install test lint format type-check clean build publish help

# デフォルトターゲット
help:
	@echo "space-syntax-analyzer 開発タスク"
	@echo ""
	@echo "利用可能なコマンド:"
	@echo "  install      - 依存関係のインストール"
	@echo "  install-dev  - 開発用依存関係のインストール"
	@echo "  test         - テストの実行"
	@echo "  test-cov     - カバレッジ付きテストの実行"
	@echo "  lint         - コードの静的解析"
	@echo "  format       - コードフォーマット"
	@echo "  type-check   - 型チェック"
	@echo "  quality      - 全品質チェック"
	@echo "  clean        - 一時ファイルの削除"
	@echo "  build        - パッケージのビルド"
	@echo "  publish-test - テストPyPIへの公開"
	@echo "  publish      - PyPIへの公開"

# パッケージ管理
install:
	uv sync

install-dev:
	uv sync --dev

# テスト
test:
	@if [ -d "space_syntax_analyzer" ]; then \
		uv run pytest; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "プロジェクト構造を確認してください"; \
		exit 1; \
	fi

test-cov:
	@if [ -d "space_syntax_analyzer" ]; then \
		uv run pytest --cov=space_syntax_analyzer --cov-report=html --cov-report=term; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		exit 1; \
	fi

# コード品質
lint:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🔍 Ruffでコードをチェック中..."; \
		uv run ruff check space_syntax_analyzer/ || echo "⚠️  Ruffチェックで警告があります"; \
		if [ -d "tests" ]; then \
			uv run ruff check tests/ || echo "⚠️  テストでRuffチェック警告があります"; \
		fi; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		exit 1; \
	fi

format:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🎨 コードをフォーマット中..."; \
		uv run black space_syntax_analyzer/; \
		uv run isort space_syntax_analyzer/; \
		if [ -d "tests" ]; then \
			uv run black tests/; \
			uv run isort tests/; \
		fi; \
		if [ -d "examples" ]; then \
			uv run black examples/; \
			uv run isort examples/; \
		fi; \
		echo "✅ フォーマット完了"; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "以下のコマンドでディレクトリを作成してください:"; \
		echo "  mkdir -p space_syntax_analyzer/core space_syntax_analyzer/utils tests examples"; \
		exit 1; \
	fi

type-check:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🔍 型チェック中..."; \
		uv run mypy space_syntax_analyzer/ || echo "⚠️  型チェックで警告があります"; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		exit 1; \
	fi

# 全品質チェック
quality: format lint type-check test
	@echo "✅ 全品質チェック完了"

# クリーンアップ
clean:
	@echo "🧹 一時ファイルを削除中..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "✅ クリーンアップ完了"

# ビルド
build: clean
	@if [ -f "pyproject.toml" ]; then \
		echo "📦 パッケージをビルド中..."; \
		uv build; \
		echo "✅ ビルド完了"; \
	else \
		echo "❌ pyproject.toml が見つかりません"; \
		exit 1; \
	fi

# PyPI公開（テスト環境）
publish-test: build
	@echo "🚀 テストPyPIに公開中..."
	uv publish --repository testpypi

# PyPI公開（本番環境）
publish: build
	@echo "🚀 PyPIに公開中..."
	uv publish

# ドキュメント生成
docs:
	@if [ -d "docs" ]; then \
		cd docs && make html; \
	else \
		echo "📚 docsディレクトリが見つかりません"; \
	fi

# 開発用サーバー起動
dev:
	@if [ -f "examples/basic_usage.py" ]; then \
		uv run python examples/basic_usage.py; \
	else \
		echo "❌ examples/basic_usage.py が見つかりません"; \
	fi

# プロジェクト初期化（新しいプロジェクト用）
init:
	@echo "🏗️  プロジェクト構造を作成中..."
	mkdir -p space_syntax_analyzer/core
	mkdir -p space_syntax_analyzer/utils
	mkdir -p tests
	mkdir -p examples
	mkdir -p docs
	touch space_syntax_analyzer/__init__.py
	touch space_syntax_analyzer/core/__init__.py
	touch space_syntax_analyzer/utils/__init__.py
	touch tests/__init__.py
	@echo "✅ プロジェクト構造作成完了"
	@echo ""
	@echo "次のステップ:"
	@echo "1. 各ファイルにコードをコピー"
	@echo "2. make install-dev"
	@echo "3. make format"

# 状態確認
status:
	@echo "📊 プロジェクト状態:"
	@echo ""
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "✅ space_syntax_analyzer/ ディレクトリ: 存在"; \
		echo "📁 ファイル数: $$(find space_syntax_analyzer -name "*.py" | wc -l)"; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリ: 不在"; \
	fi
	@if [ -f "pyproject.toml" ]; then \
		echo "✅ pyproject.toml: 存在"; \
	else \
		echo "❌ pyproject.toml: 不在"; \
	fi
	@if [ -d "tests" ]; then \
		echo "✅ tests/ ディレクトリ: 存在"; \
		echo "📁 テストファイル数: $$(find tests -name "test_*.py" | wc -l)"; \
	else \
		echo "❌ tests/ ディレクトリ: 不在"; \
	fi