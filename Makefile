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
	@echo "ℹ️  TestPyPIのAPIトークン認証が必要です"
	@if [ -z "$TWINE_PASSWORD" ] && [ ! -f ~/.pypirc ]; then \
		echo "❌ 認証情報が設定されていません"; \
		echo ""; \
		echo "以下のいずれかの方法で設定してください:"; \
		echo ""; \
		echo "方法1: 環境変数"; \
		echo "  export TWINE_USERNAME=__token__"; \
		echo "  export TWINE_PASSWORD=pypi-your-testpypi-api-token"; \
		echo "  export TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/"; \
		echo ""; \
		echo "方法2: ~/.pypirc ファイル作成"; \
		echo "  詳細は https://test.pypi.org/help/#apitoken を参照"; \
		echo ""; \
		exit 1; \
	fi

	@if ! uv run twine --version >/dev/null 2>&1; then \
		echo "❌ twineが見つかりません。twineをインストールしてください:"; \
		echo "  uv add --dev twine"; \
		exit 1; \
	fi
	@echo "📤 twineを使用してTestPyPIに公開..."; \
	echo "🔑 ~/.pypirc の [testpypi] セクションを使用"; \
	uv run twine upload --repository testpypi dist/*

# PyPI公開（本番環境）
publish: build
	@echo "🚀 PyPIに公開中..."
	@echo "⚠️  本番環境への公開です。実行前に確認してください"
	@if [ ! -f ~/.pypirc ]; then \
		echo "❌ ~/.pypirc ファイルが見つかりません"; \
		echo "詳細は publish-test ターゲットのヘルプを参照してください"; \
		exit 1; \
	fi
	@if ! grep -q "\[pypi\]" ~/.pypirc; then \
		echo "❌ ~/.pypirc に [pypi] セクションが見つかりません"; \
		echo "ℹ️  ファイルを確認してください: cat ~/.pypirc"; \
		exit 1; \
	fi
	@if ! uv run twine --version >/dev/null 2>&1; then \
		echo "❌ twineが見つかりません。twineをインストールしてください:"; \
		echo "  uv add --dev twine"; \
		exit 1; \
	fi
	@echo "続行するには 'yes' を入力してください:"
	@read confirmation; \
	if [ "$$confirmation" = "yes" ]; then \
		echo "📤 twineを使用してPyPIに公開..."; \
		echo "🔑 ~/.pypirc の [pypi] セクションを使用"; \
		uv run twine upload --repository pypi dist/*; \
	else \
		echo "❌ 公開をキャンセルしました"; \
	fi

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

# デバッグ用: 認証情報の確認
# デバッグ用: 認証情報の確認
check-auth:
	@echo "🔍 認証情報を確認中..."
	@if [ -n "$$TWINE_PASSWORD" ]; then \
		echo "✅ TWINE_PASSWORD: 設定済み (環境変数)"; \
	else \
		echo "ℹ️  TWINE_PASSWORD: 未設定 (環境変数)"; \
	fi
	@if [ -n "$$TWINE_USERNAME" ]; then \
		echo "✅ TWINE_USERNAME: $$TWINE_USERNAME (環境変数)"; \
	else \
		echo "ℹ️  TWINE_USERNAME: 未設定 (環境変数)"; \
	fi
	@if [ -f ~/.pypirc ]; then \
		echo "✅ ~/.pypirc: 存在"; \
		echo ""; \
		echo "📄 ファイル内容:"; \
		echo "--- ~/.pypirc ---"; \
		cat ~/.pypirc; \
		echo "--- end ---"; \
		echo ""; \
		if grep -q "\[testpypi\]" ~/.pypirc; then \
			echo "✅ [testpypi] セクション: 存在"; \
		else \
			echo "❌ [testpypi] セクション: 不在"; \
		fi; \
		if grep -q "\[pypi\]" ~/.pypirc; then \
			echo "✅ [pypi] セクション: 存在"; \
		else \
			echo "❌ [pypi] セクション: 不在"; \
		fi; \
		if grep -q "username = __token__" ~/.pypirc; then \
			echo "✅ username設定: __token__ を使用"; \
		else \
			echo "⚠️  username設定: __token__ ではない可能性"; \
		fi; \
		if grep -q "password = pypi-" ~/.pypirc; then \
			echo "✅ password設定: APIトークン形式"; \
		else \
			echo "⚠️  password設定: APIトークンではない可能性"; \
		fi; \
	else \
		echo "❌ ~/.pypirc: 不在"; \
		echo ""; \
		echo "作成方法:"; \
		echo "  make setup-pypirc"; \
	fi
	@if uv run twine --version >/dev/null 2>&1; then \
		echo "✅ twine: インストール済み (uv環境)"; \
		echo "   バージョン: $$(uv run twine --version 2>/dev/null)"; \
	else \
		echo "❌ twine: 未インストール"; \
		echo "  インストール: uv add --dev twine"; \
	fi

# テスト用: 認証なしでビルドのみ
test-build: build
	@echo "✅ ビルドテスト完了"
	@echo "📁 作成されたファイル:"
	@ls -la dist/
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

# 依存関係の確認
check-deps:
	@echo "🔍 依存関係を確認中..."
	@uv tree || echo "⚠️  uvの依存関係ツリー表示でエラーが発生しました"

# セキュリティチェック
security:
	@echo "🔒 セキュリティチェック中..."
	@if command -v safety >/dev/null 2>&1; then \
		uv run safety check; \
	else \
		echo "⚠️  safetyが見つかりません。インストールするには:"; \
		echo "  uv add --dev safety"; \
	fi