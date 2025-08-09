.PHONY: install test lint format type-check clean build publish help init-project

# デフォルトターゲット
help:
	@echo "space-syntax-analyzer 開発タスク"
	@echo ""
	@echo "利用可能なコマンド:"
	@echo "  init-project - プロジェクト構造の初期化"
	@echo "  install      - 依存関係のインストール"
	@echo "  install-dev  - 開発用依存関係のインストール"
	@echo "  test         - テストの実行"
	@echo "  test-cov     - カバレッジ付きテストの実行"
	@echo "  lint         - コードの静的解析"
	@echo "  lint-fix     - 自動修正可能な問題の修正"
	@echo "  format       - コードフォーマット"
	@echo "  type-check   - 型チェック"
	@echo "  quality      - 全品質チェック"
	@echo "  clean        - 一時ファイルの削除"
	@echo "  build        - パッケージのビルド"
	@echo "  demo         - デモ実行"
	@echo "  status       - プロジェクト状態確認"
	@echo "  diagnose     - 詳細診断"
	@echo "  publish-test - テストPyPIへの公開"
	@echo "  publish      - PyPIへの公開"

# プロジェクト構造の初期化
init-project:
	@echo "🏗️  プロジェクト構造を作成中..."
	@mkdir -p space_syntax_analyzer/core
	@mkdir -p space_syntax_analyzer/utils
	@mkdir -p space_syntax_analyzer/examples
	@mkdir -p tests
	@mkdir -p examples
	@mkdir -p docs
	@mkdir -p demo_output
	
	@# 基本__init__.pyファイルの作成
	@echo '"""Space Syntax Analyzer - スペースシンタックス分析ライブラリ"""' > space_syntax_analyzer/__init__.py
	@echo '' >> space_syntax_analyzer/__init__.py
	@echo '__version__ = "0.2.0"' >> space_syntax_analyzer/__init__.py
	@echo '__author__ = "Space Syntax Analyzer Team"' >> space_syntax_analyzer/__init__.py
	@echo '' >> space_syntax_analyzer/__init__.py
	@echo '# 基本的なインポート' >> space_syntax_analyzer/__init__.py
	@echo 'try:' >> space_syntax_analyzer/__init__.py
	@echo '    from .core.analyzer import SpaceSyntaxAnalyzer' >> space_syntax_analyzer/__init__.py
	@echo '    __all__ = ["SpaceSyntaxAnalyzer"]' >> space_syntax_analyzer/__init__.py
	@echo 'except ImportError:' >> space_syntax_analyzer/__init__.py
	@echo '    # 開発時のフォールバック' >> space_syntax_analyzer/__init__.py
	@echo '    __all__ = []' >> space_syntax_analyzer/__init__.py
	
	@echo '"""コアモジュール"""' > space_syntax_analyzer/core/__init__.py
	@echo '"""ユーティリティモジュール"""' > space_syntax_analyzer/utils/__init__.py
	@echo '"""使用例モジュール"""' > space_syntax_analyzer/examples/__init__.py
	@echo '"""テストモジュール"""' > tests/__init__.py
	
	@# README.mdの作成
	@if [ ! -f "README.md" ]; then \
		echo "# space-syntax-analyzer" > README.md; \
		echo "" >> README.md; \
		echo "スペースシンタックス理論に基づく都市空間分析ライブラリ" >> README.md; \
		echo "" >> README.md; \
		echo "## インストール" >> README.md; \
		echo "" >> README.md; \
		echo "\`\`\`bash" >> README.md; \
		echo "pip install space-syntax-analyzer" >> README.md; \
		echo "\`\`\`" >> README.md; \
		echo "" >> README.md; \
		echo "## 基本的な使用方法" >> README.md; \
		echo "" >> README.md; \
		echo "\`\`\`python" >> README.md; \
		echo "from space_syntax_analyzer import SpaceSyntaxAnalyzer" >> README.md; \
		echo "" >> README.md; \
		echo "analyzer = SpaceSyntaxAnalyzer()" >> README.md; \
		echo "results = analyzer.analyze_place('Tokyo, Japan')" >> README.md; \
		echo "\`\`\`" >> README.md; \
	fi
	
	@echo "✅ プロジェクト構造作成完了"
	@echo ""
	@echo "作成されたディレクトリ:"
	@echo "  space_syntax_analyzer/core/     - コア機能"
	@echo "  space_syntax_analyzer/utils/    - ユーティリティ"
	@echo "  space_syntax_analyzer/examples/ - 使用例"
	@echo "  tests/                          - テスト"
	@echo "  examples/                       - 外部例"
	@echo "  docs/                           - ドキュメント"
	@echo "  demo_output/                    - デモ出力"
	@echo ""
	@echo "次のステップ:"
	@echo "1. 各ファイルにコードをコピー"
	@echo "2. make install-dev"
	@echo "3. make format"

# パッケージ管理
install:
	@echo "📦 依存関係をインストール中..."
	uv sync

install-dev:
	@echo "🔧 開発用依存関係をインストール中..."
	uv sync --all-extras

# テスト
test:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🧪 テスト実行中..."; \
		uv run pytest; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "プロジェクト構造を作成するには: make init-project"; \
		exit 1; \
	fi

test-cov:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "📊 カバレッジ付きテスト実行中..."; \
		uv run pytest --cov=space_syntax_analyzer --cov-report=html --cov-report=term; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "プロジェクト構造を作成するには: make init-project"; \
		exit 1; \
	fi

test-unit:
	@echo "⚡ 単体テスト実行中..."
	@uv run pytest -m "unit or not integration" -v

test-integration:
	@echo "🔗 統合テスト実行中..."
	@uv run pytest -m integration -v

test-lightweight:
	@echo "🪶 軽量テスト実行中..."
	@if [ -f "tests/test_analyzer_lightweight.py" ]; then \
		uv run python tests/test_analyzer_lightweight.py; \
	else \
		echo "⚠️  軽量テストファイルが見つかりません"; \
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
		echo "プロジェクト構造を作成するには: make init-project"; \
		exit 1; \
	fi

lint-fix:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🔧 自動修正可能な問題を修正中..."; \
		uv run ruff check space_syntax_analyzer/ --fix --unsafe-fixes; \
		if [ -d "tests" ]; then \
			uv run ruff check tests/ --fix --unsafe-fixes; \
		fi; \
		echo "✅ 自動修正完了"; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "プロジェクト構造を作成するには: make init-project"; \
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
		echo "プロジェクト構造を作成するには: make init-project"; \
		exit 1; \
	fi

type-check:
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "🔍 型チェック中..."; \
		uv run mypy space_syntax_analyzer/ || echo "⚠️  型チェックで警告があります"; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリが見つかりません"; \
		echo "プロジェクト構造を作成するには: make init-project"; \
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
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	@echo "✅ クリーンアップ完了"

# ビルド
build: clean
	@if [ -f "pyproject.toml" ]; then \
		echo "📦 パッケージをビルド中..."; \
		uv build; \
		echo "✅ ビルド完了"; \
		echo "📁 作成されたファイル:"; \
		ls -la dist/; \
	else \
		echo "❌ pyproject.toml が見つかりません"; \
		exit 1; \
	fi

# デモ実行
demo:
	@if [ -f "examples/demo.py" ]; then \
		echo "🚀 デモ実行中..."; \
		mkdir -p demo_output; \
		uv run python examples/demo.py; \
	elif [ -f "space_syntax_analyzer/examples/demo.py" ]; then \
		echo "🚀 デモ実行中..."; \
		mkdir -p demo_output; \
		uv run python space_syntax_analyzer/examples/demo.py; \
	else \
		echo "❌ デモファイルが見つかりません"; \
		echo "以下のいずれかの場所にdemo.pyを配置してください:"; \
		echo "  - examples/demo.py"; \
		echo "  - space_syntax_analyzer/examples/demo.py"; \
	fi

demo-auto:
	@echo "🤖 自動デモ実行中..."
	@mkdir -p demo_output
	@DEMO_AUTO_MODE=true make demo

# 開発サーバー（基本的な動作確認）
dev:
	@echo "🔧 開発モード: 基本動作確認"
	@if [ -f "tests/test_analyzer_lightweight.py" ]; then \
		echo "軽量テストで動作確認中..."; \
		uv run python tests/test_analyzer_lightweight.py; \
	else \
		echo "基本インポートテスト中..."; \
		uv run python -c "import space_syntax_analyzer; print('✅ インポート成功')"; \
	fi

# PyPI公開（テスト環境）
publish-test: build
	@echo "🚀 テストPyPIに公開中..."
	@echo "ℹ️  TestPyPIのAPIトークン認証が必要です"
	@if [ -z "$$TWINE_PASSWORD" ] && [ ! -f ~/.pypirc ]; then \
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
		echo "📖 ドキュメント生成中..."; \
		cd docs && make html; \
	else \
		echo "📚 docsディレクトリが見つかりません"; \
		echo "ドキュメント構造を作成してください"; \
	fi

docs-serve:
	@if [ -d "docs/_build/html" ]; then \
		echo "🌐 ドキュメントサーバー起動中..."; \
		cd docs/_build/html && python -m http.server 8000; \
	else \
		echo "❌ ドキュメントが生成されていません"; \
		echo "先に 'make docs' を実行してください"; \
	fi

# 依存関係の確認
check-deps:
	@echo "🔍 依存関係を確認中..."
	@uv tree || echo "⚠️  uvの依存関係ツリー表示でエラーが発生しました"

# プロジェクト状態の確認
status:
	@echo "📊 プロジェクト状態:"
	@echo ""
	@echo "📁 ディレクトリ構造:"
	@if [ -d "space_syntax_analyzer" ]; then \
		echo "✅ space_syntax_analyzer/ ディレクトリ: 存在"; \
		echo "📁 Pythonファイル数: $(find space_syntax_analyzer -name "*.py" 2>/dev/null | wc -l)"; \
		if [ -f "space_syntax_analyzer/__init__.py" ]; then \
			echo "✅ space_syntax_analyzer/__init__.py: 存在"; \
		else \
			echo "❌ space_syntax_analyzer/__init__.py: 不在"; \
		fi; \
		if [ -d "space_syntax_analyzer/core" ]; then \
			echo "✅ space_syntax_analyzer/core/: 存在"; \
		else \
			echo "❌ space_syntax_analyzer/core/: 不在"; \
		fi; \
		if [ -d "space_syntax_analyzer/utils" ]; then \
			echo "✅ space_syntax_analyzer/utils/: 存在"; \
		else \
			echo "❌ space_syntax_analyzer/utils/: 不在"; \
		fi; \
	else \
		echo "❌ space_syntax_analyzer/ ディレクトリ: 不在"; \
		echo "   → make init-project で作成してください"; \
	fi
	@echo ""
	@echo "📄 設定ファイル:"
	@if [ -f "pyproject.toml" ]; then \
		echo "✅ pyproject.toml: 存在"; \
	else \
		echo "❌ pyproject.toml: 不在"; \
	fi
	@if [ -f "README.md" ]; then \
		echo "✅ README.md: 存在"; \
	else \
		echo "❌ README.md: 不在"; \
	fi
	@echo ""
	@echo "🧪 テスト:"
	@if [ -d "tests" ]; then \
		echo "✅ tests/ ディレクトリ: 存在"; \
		echo "📁 テストファイル数: $(find tests -name "test_*.py" 2>/dev/null | wc -l)"; \
	else \
		echo "❌ tests/ ディレクトリ: 不在"; \
	fi
	@echo ""
	@echo "🐍 Python環境:"
	@if [ -d ".venv" ]; then \
		echo "✅ 仮想環境: 存在"; \
	else \
		echo "❌ 仮想環境: 不在"; \
		echo "   → uv sync で作成してください"; \
	fi
	@echo ""
	@echo "📦 ビルドテスト:"
	@python -c "import toml; data=toml.load('pyproject.toml'); print('✅ pyproject.toml 構文: 正常')" 2>/dev/null || echo "❌ pyproject.toml 構文: エラー"

# 診断とトラブルシューティング
diagnose:
	@echo "🔍 詳細診断実行中..."
	@echo ""
	@echo "=== 現在のディレクトリ ==="
	@pwd
	@echo ""
	@echo "=== ディレクトリ内容 ==="
	@ls -la
	@echo ""
	@echo "=== space_syntax_analyzer/ 内容 ==="
	@if [ -d "space_syntax_analyzer" ]; then \
		find space_syntax_analyzer -type f -name "*.py" | head -10; \
	else \
		echo "space_syntax_analyzer/ ディレクトリが存在しません"; \
	fi
	@echo ""
	@echo "=== pyproject.toml 検証 ==="
	@if [ -f "pyproject.toml" ]; then \
		echo "pyproject.toml が存在します"; \
		echo ""; \
		echo "--- バージョン確認 ---"; \
		grep "version" pyproject.toml || echo "バージョンが見つかりません"; \
		echo ""; \
		echo "--- 依存関係確認 ---"; \
		grep -A 5 "dependencies" pyproject.toml || echo "依存関係が見つかりません"; \
		echo ""; \
		echo "--- 分類子確認 ---"; \
		grep -A 10 "classifiers" pyproject.toml || echo "分類子が見つかりません"; \
	else \
		echo "pyproject.toml が存在しません"; \
	fi
	@echo ""
	@echo "=== Python/UV 環境 ==="
	@python --version || echo "Python が見つかりません"
	@uv --version || echo "UV が見つかりません"
	@if [ -d ".venv" ]; then \
		echo ".venv ディレクトリが存在します"; \
	else \
		echo ".venv ディレクトリが存在しません"; \
	fi

# 開発環境のセットアップ（ワンストップ）
setup: init-project install-dev
	@echo "🎉 開発環境セットアップ完了！"
	@echo ""
	@echo "次のステップ:"
	@echo "1. 各ファイルにコードをコピー"
	@echo "2. make format"
	@echo "3. make test-lightweight"
	@echo "4. make demo"

# セキュリティチェック
security:
	@echo "🔒 セキュリティチェック中..."
	@if command -v safety >/dev/null 2>&1; then \
		uv run safety check; \
	else \
		echo "⚠️  safetyが見つかりません。インストールするには:"; \
		echo "  uv add --dev safety"; \
	fi

# 認証情報確認（デバッグ用）
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
	else \
		echo "❌ ~/.pypirc: 不在"; \
	fi
	@if uv run twine --version >/dev/null 2>&1; then \
		echo "✅ twine: インストール済み"; \
	else \
		echo "❌ twine: 未インストール"; \
	fi