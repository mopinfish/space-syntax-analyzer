# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx-Makefile using the "make mode" option.
# $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# カスタムターゲット
clean:
	@echo "🧹 ドキュメントビルドファイルを削除中..."
	rm -rf $(BUILDDIR)/*
	@echo "✅ クリーンアップ完了"

html-watch:
	@echo "👀 ドキュメントの自動リビルドを開始..."
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)

livehtml: html-watch

api:
	@echo "📖 API ドキュメントを生成中..."
	sphinx-apidoc -o $(SOURCEDIR)/api ../space_syntax_analyzer --force --module-first
	@echo "✅ API ドキュメント生成完了"

html-ja:
	@echo "🌸 日本語ドキュメントをビルド中..."
	$(SPHINXBUILD) -b html -D language=ja $(SOURCEDIR) $(BUILDDIR)/html-ja $(SPHINXOPTS) $(O)
	@echo "✅ 日本語ドキュメントビルド完了"

html-en:
	@echo "🇺🇸 英語ドキュメントをビルド中..."
	$(SPHINXBUILD) -b html -D language=en $(SOURCEDIR) $(BUILDDIR)/html-en $(SPHINXOPTS) $(O)
	@echo "✅ 英語ドキュメントビルド完了"

multihtml: html-ja html-en
	@echo "✅ 多言語ドキュメントビルド完了"

linkcheck:
	@echo "🔗 リンクチェック中..."
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS) $(O)

coverage:
	@echo "📊 ドキュメントカバレッジ計測中..."
	$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage $(SPHINXOPTS) $(O)

doctest:
	@echo "🧪 ドキュメント内のコードテスト中..."
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS) $(O)

pdf:
	@echo "📄 PDFドキュメント生成中..."
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS) $(O)
	@make -C $(BUILDDIR)/latex all-pdf

publish:
	@echo "🚀 ドキュメントを公開用ディレクトリにコピー中..."
	@if [ -d "$(BUILDDIR)/html" ]; then \
		cp -r $(BUILDDIR)/html/* ../public/ 2>/dev/null || mkdir -p ../public && cp -r $(BUILDDIR)/html/* ../public/; \
		echo "✅ ドキュメント公開準備完了"; \
	else \
		echo "❌ HTMLドキュメントが見つかりません。先に 'make html' を実行してください"; \
	fi

serve:
	@echo "🌐 ローカルサーバーを起動中..."
	@if [ -d "$(BUILDDIR)/html" ]; then \
		cd $(BUILDDIR)/html && python -m http.server 8000; \
	else \
		echo "❌ HTMLドキュメントが見つかりません。先に 'make html' を実行してください"; \
	fi

dev: clean api html serve

# ヘルプの拡張
help-extended:
	@echo "space-syntax-analyzer ドキュメント Makefile"
	@echo ""
	@echo "標準ターゲット:"
	@echo "  html         HTMLドキュメントをビルド"
	@echo "  clean        ビルドファイルを削除"
	@echo "  linkcheck    外部リンクをチェック"
	@echo "  doctest      ドキュメント内のコードをテスト"
	@echo ""
	@echo "カスタムターゲット:"
	@echo "  api          API ドキュメントを自動生成"
	@echo "  html-ja      日本語版HTMLをビルド"
	@echo "  html-en      英語版HTMLをビルド"
	@echo "  multihtml    多言語版HTMLをビルド"
	@echo "  html-watch   自動リビルド付きでHTMLをビルド"
	@echo "  coverage     ドキュメントカバレッジを計測"
	@echo "  pdf          PDFドキュメントを生成"
	@echo "  serve        ローカルサーバーでドキュメントを表示"
	@echo "  publish      ドキュメントを公開用ディレクトリにコピー"
	@echo "  dev          開発用（clean + api + html + serve）"
	@echo ""
	@echo "例:"
	@echo "  make html                # 基本的なHTMLビルド"
	@echo "  make html-watch          # 自動リビルド付きビルド"
	@echo "  make dev                 # 開発用ワンストップコマンド"