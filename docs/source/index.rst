space-syntax-analyzer ドキュメント
=====================================

**space-syntax-analyzer** は、スペースシンタックス理論に基づいた都市空間の分析を行うPythonライブラリです。

OpenStreetMapの道路ネットワークデータを使用して、都市の空間構造を定量的に分析・可視化できます。

特徴
----

* **簡単な API**: 3行のコードで基本分析を実行
* **スペースシンタックス理論**: Bill Hillierらの理論に基づく分析
* **OSMnx統合**: リアルタイムな地理空間データ取得
* **豊富な可視化**: matplotlib基盤の高品質な図表生成
* **多様な出力形式**: CSV、Excel、JSON、GeoJSON対応

クイックスタート
----------------

.. code-block:: python

   from space_syntax_analyzer import SpaceSyntaxAnalyzer

   # アナライザーの初期化
   analyzer = SpaceSyntaxAnalyzer()

   # 渋谷駅周辺の分析
   results = analyzer.analyze_place("Shibuya, Tokyo, Japan")

   # 結果の表示
   print(analyzer.generate_report(results, "渋谷駅周辺"))

インストール
------------

.. code-block:: bash

   # uvを使用（推奨）
   uv add space-syntax-analyzer

   # pipを使用
   pip install space-syntax-analyzer

目次
----

.. toctree::
   :maxdepth: 2
   :caption: ユーザーガイド:

   installation
   quickstart
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: API リファレンス:

   api/modules
   api/space_syntax_analyzer

.. toctree::
   :maxdepth: 1
   :caption: 開発者向け:

   development
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: その他:

   license
   citing

索引とテーブル
==============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`