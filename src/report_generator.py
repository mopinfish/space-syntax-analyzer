"""
レポート生成モジュール
パス: src/report_generator.py

Space Syntax解析結果の総合レポート生成
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .config_manager import ReportSettings


class ReportGenerator:
    """レポート生成クラス"""
    
    def __init__(self, settings: dict):
        """
        初期化
        
        Args:
            settings: レポート設定辞書
        """
        self.logger = logging.getLogger(__name__)
        self.settings = ReportSettings(**settings)
        self.logger.info("ReportGenerator初期化完了")
    
    def generate_comprehensive_report(self, results: Dict[str, Dict], 
                                    location_params: Dict[str, Any],
                                    output_path: Path):
        """
        総合レポートの生成
        
        Args:
            results: 解析結果辞書
            location_params: 地域パラメータ
            output_path: 出力パス
        """
        self.logger.info("総合レポート生成開始")
        
        try:
            # HTMLレポート生成
            html_path = output_path.with_suffix('.html')
            self._generate_html_report(results, location_params, html_path)
            
            # PDFレポート生成（HTMLから変換）
            self._convert_html_to_pdf(html_path, output_path)
            
            self.logger.info(f"総合レポート生成完了: {output_path}")
            
        except Exception as e:
            self.logger.error(f"総合レポート生成エラー: {e}")
            raise
    
    def _generate_html_report(self, results: Dict[str, Dict], 
                             location_params: Dict[str, Any],
                             output_path: Path):
        """
        HTMLレポートの生成
        
        Args:
            results: 解析結果辞書
            location_params: 地域パラメータ
            output_path: 出力パス
        """
        try:
            # HTML構造の構築
            html_content = self._build_html_structure(results, location_params)
            
            # HTMLファイル書き込み
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")
            raise
    
    def _build_html_structure(self, results: Dict[str, Dict], 
                             location_params: Dict[str, Any]) -> str:
        """
        HTML構造の構築
        
        Args:
            results: 解析結果辞書
            location_params: 地域パラメータ
            
        Returns:
            HTML文字列
        """
        try:
            # CSSスタイル
            css_style = self._get_css_style()
            
            # ヘッダー情報
            header_info = self._build_header_section(location_params)
            
            # エグゼクティブサマリー
            executive_summary = self._build_executive_summary(results)
            
            # 解析結果詳細
            analysis_details = self._build_analysis_details(results)
            
            # 統計表
            statistics_tables = self._build_statistics_tables(results)
            
            # 相関分析
            correlation_analysis = self._build_correlation_analysis(results)
            
            # 結論・提言
            conclusions = self._build_conclusions(results)
            
            # 付録
            appendix = self._build_appendix()
            
            # 完全なHTML構築
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Space Syntax解析レポート</title>
    <style>{css_style}</style>
</head>
<body>
    {header_info}
    {executive_summary}
    {analysis_details}
    {statistics_tables}
    {correlation_analysis}
    {conclusions}
    {appendix}
    
    <footer>
        <p>このレポートは Space Syntax解析システム により自動生成されました。</p>
        <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML構造構築エラー: {e}")
            raise
    
    def _get_css_style(self) -> str:
        """CSSスタイルの取得"""
        return """
        body {
            font-family: 'Hiragino Sans', 'Yu Gothic Medium', 'Meiryo', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2E7D32;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #4CAF50;
            background-color: #f9f9f9;
        }
        .section h2 {
            color: #2E7D32;
            margin-top: 0;
            font-size: 1.8em;
        }
        .section h3 {
            color: #388E3C;
            font-size: 1.4em;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .info-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .info-card h4 {
            margin: 0 0 10px 0;
            color: #2E7D32;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .metrics-table th,
        .metrics-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .metrics-table th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .metrics-table tr:hover {
            background-color: #e8f5e8;
        }
        .highlight {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .warning {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .correlation-matrix {
            display: grid;
            gap: 10px;
            margin: 20px 0;
        }
        .correlation-row {
            display: grid;
            grid-template-columns: 150px repeat(auto-fit, minmax(100px, 1fr));
            gap: 5px;
            align-items: center;
        }
        .correlation-cell {
            padding: 8px;
            text-align: center;
            border-radius: 4px;
            font-weight: bold;
        }
        .correlation-strong { background-color: #c8e6c9; }
        .correlation-moderate { background-color: #fff9c4; }
        .correlation-weak { background-color: #ffcdd2; }
        .toc {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .toc a {
            text-decoration: none;
            color: #2E7D32;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }
        @media print {
            body { font-size: 12px; }
            .section { page-break-inside: avoid; }
        }
        """
    
    def _build_header_section(self, location_params: Dict[str, Any]) -> str:
        """ヘッダーセクションの構築"""
        try:
            location_info = ""
            if 'place' in location_params:
                location_info = f"解析対象地域: {location_params['place']}"
            elif 'bbox' in location_params:
                bbox = location_params['bbox']
                location_info = f"解析対象範囲: ({bbox[1]:.4f}, {bbox[0]:.4f}) - ({bbox[3]:.4f}, {bbox[2]:.4f})"
            elif 'admin' in location_params:
                location_info = f"解析対象行政区域: {location_params['admin']}"
            
            return f"""
            <div class="container">
                <div class="header">
                    <h1>Space Syntax解析レポート</h1>
                    <div class="subtitle">{location_info}</div>
                    <div class="subtitle">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
                </div>
                
                <div class="toc">
                    <h3>目次</h3>
                    <ul>
                        <li><a href="#executive-summary">1. エグゼクティブサマリー</a></li>
                        <li><a href="#analysis-details">2. 解析結果詳細</a></li>
                        <li><a href="#statistics">3. 統計分析</a></li>
                        <li><a href="#correlations">4. 相関分析</a></li>
                        <li><a href="#conclusions">5. 結論・提言</a></li>
                        <li><a href="#appendix">6. 付録</a></li>
                    </ul>
                </div>
            """
            
        except Exception as e:
            self.logger.error(f"ヘッダーセクション構築エラー: {e}")
            return "<div class='container'><div class='header'><h1>Space Syntax解析レポート</h1></div>"
    
    def _build_executive_summary(self, results: Dict[str, Dict]) -> str:
        """エグゼクティブサマリーの構築"""
        try:
            summary_cards = []
            
            for analysis_type, data in results.items():
                if 'metrics' not in data:
                    continue
                
                metrics = data['metrics']
                graph = data.get('graph')
                
                # 基本情報
                node_count = len(graph.nodes()) if graph else 0
                edge_count = len(graph.edges()) if graph else 0
                
                # 主要指標の統計
                integration_stats = ""
                connectivity_stats = ""
                choice_stats = ""
                
                if 'integration' in metrics:
                    integ_series = metrics['integration']
                    integration_stats = f"平均: {integ_series.mean():.3f}, 標準偏差: {integ_series.std():.3f}"
                
                if 'connectivity' in metrics:
                    conn_series = metrics['connectivity']
                    connectivity_stats = f"平均: {conn_series.mean():.3f}, 最大: {conn_series.max():.0f}"
                
                if 'choice' in metrics:
                    choice_series = metrics['choice']
                    choice_stats = f"平均: {choice_series.mean():.3f}, 標準偏差: {choice_series.std():.3f}"
                
                summary_cards.append(f"""
                <div class="info-card">
                    <h4>{analysis_type.upper()} MAP解析</h4>
                    <p><strong>ネットワーク規模:</strong> {node_count:,}ノード, {edge_count:,}エッジ</p>
                    {f'<p><strong>Integration:</strong> {integration_stats}</p>' if integration_stats else ''}
                    {f'<p><strong>Connectivity:</strong> {connectivity_stats}</p>' if connectivity_stats else ''}
                    {f'<p><strong>Choice:</strong> {choice_stats}</p>' if choice_stats else ''}
                </div>
                """)
            
            # Intelligibility分析
            intelligibility_summary = ""
            for analysis_type, data in results.items():
                if 'intelligibility' in data.get('metrics', {}):
                    intell_value = data['metrics']['intelligibility']
                    if isinstance(intell_value, (int, float)):
                        intelligibility_summary += f"""
                        <div class="highlight">
                            <h4>{analysis_type.upper()} MAP Intelligibility</h4>
                            <p>相関係数: <span class="stat-value">{intell_value:.3f}</span></p>
                            <p>{self._interpret_intelligibility(intell_value)}</p>
                        </div>
                        """
            
            return f"""
            <div class="section" id="executive-summary">
                <h2>1. エグゼクティブサマリー</h2>
                
                <h3>解析概要</h3>
                <p>本レポートでは、Space Syntax理論に基づく都市空間構造の定量分析を実施しました。
                解析では以下の主要指標を計算し、空間の特性を評価しています。</p>
                
                <div class="info-grid">
                    {''.join(summary_cards)}
                </div>
                
                {intelligibility_summary}
                
                <div class="highlight">
                    <h4>主要な知見</h4>
                    <ul>
                        <li>ネットワークの接続性パターンが明らかになりました</li>
                        <li>空間統合性の分布特性を定量化しました</li>
                        <li>経路選択性の高い軸線・セグメントを特定しました</li>
                        <li>空間認知性（Intelligibility）の水準を評価しました</li>
                    </ul>
                </div>
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"エグゼクティブサマリー構築エラー: {e}")
            return '<div class="section"><h2>1. エグゼクティブサマリー</h2><p>エラーが発生しました。</p></div>'
    
    def _build_analysis_details(self, results: Dict[str, Dict]) -> str:
        """解析結果詳細の構築"""
        try:
            details_sections = []
            
            for analysis_type, data in results.items():
                if 'metrics' not in data:
                    continue
                
                metrics = data['metrics']
                
                # 指標説明
                metric_descriptions = self._get_metric_descriptions()
                
                # 指標別詳細
                metric_details = []
                for metric_name, series in metrics.items():
                    if isinstance(series, pd.Series) and len(series) > 0:
                        desc = metric_descriptions.get(metric_name.split('_')[0], '')
                        metric_details.append(f"""
                        <h4>{self._get_metric_label(metric_name)}</h4>
                        <p>{desc}</p>
                        <ul>
                            <li>データ数: {len(series):,}</li>
                            <li>平均値: {series.mean():.4f}</li>
                            <li>標準偏差: {series.std():.4f}</li>
                            <li>範囲: {series.min():.4f} - {series.max():.4f}</li>
                            <li>中央値: {series.median():.4f}</li>
                        </ul>
                        """)
                
                details_sections.append(f"""
                <h3>{analysis_type.upper()} MAP解析結果</h3>
                <p>{self._get_analysis_description(analysis_type)}</p>
                {''.join(metric_details)}
                """)
            
            return f"""
            <div class="section" id="analysis-details">
                <h2>2. 解析結果詳細</h2>
                {''.join(details_sections)}
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"解析結果詳細構築エラー: {e}")
            return '<div class="section"><h2>2. 解析結果詳細</h2><p>エラーが発生しました。</p></div>'
    
    def _build_statistics_tables(self, results: Dict[str, Dict]) -> str:
        """統計表の構築"""
        try:
            tables = []
            
            for analysis_type, data in results.items():
                if 'statistics' not in data:
                    continue
                
                statistics = data['statistics']
                
                # 統計表のHTML作成
                table_rows = []
                for metric_name, stats in statistics.items():
                    if isinstance(stats, dict):
                        table_rows.append(f"""
                        <tr>
                            <td>{self._get_metric_label(metric_name)}</td>
                            <td>{stats.get('count', 0):,}</td>
                            <td>{stats.get('mean', 0):.4f}</td>
                            <td>{stats.get('std', 0):.4f}</td>
                            <td>{stats.get('min', 0):.4f}</td>
                            <td>{stats.get('median', 0):.4f}</td>
                            <td>{stats.get('max', 0):.4f}</td>
                            <td>{stats.get('skewness', 0):.3f}</td>
                            <td>{stats.get('kurtosis', 0):.3f}</td>
                        </tr>
                        """)
                
                if table_rows:
                    tables.append(f"""
                    <h3>{analysis_type.upper()} MAP統計表</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>指標</th>
                                <th>データ数</th>
                                <th>平均</th>
                                <th>標準偏差</th>
                                <th>最小値</th>
                                <th>中央値</th>
                                <th>最大値</th>
                                <th>歪度</th>
                                <th>尖度</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(table_rows)}
                        </tbody>
                    </table>
                    """)
            
            return f"""
            <div class="section" id="statistics">
                <h2>3. 統計分析</h2>
                <p>各Space Syntax指標の基本統計量を以下に示します。</p>
                {''.join(tables)}
                
                <div class="highlight">
                    <h4>統計値の解釈</h4>
                    <ul>
                        <li><strong>歪度</strong>: 分布の非対称性を示す（0に近いほど対称）</li>
                        <li><strong>尖度</strong>: 分布の尖り具合を示す（正の値で尖った分布）</li>
                        <li><strong>標準偏差</strong>: データのばらつきの程度</li>
                    </ul>
                </div>
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"統計表構築エラー: {e}")
            return '<div class="section"><h2>3. 統計分析</h2><p>エラーが発生しました。</p></div>'
    
    def _build_correlation_analysis(self, results: Dict[str, Dict]) -> str:
        """相関分析の構築"""
        try:
            correlation_sections = []
            
            for analysis_type, data in results.items():
                if 'correlations' not in data:
                    continue
                
                correlations = data['correlations']
                
                if isinstance(correlations, pd.DataFrame) and not correlations.empty:
                    # 相関行列のHTML表作成
                    correlation_matrix = self._build_correlation_matrix_html(correlations)
                    
                    # 強い相関の分析
                    strong_correlations = self._analyze_strong_correlations(correlations)
                    
                    correlation_sections.append(f"""
                    <h3>{analysis_type.upper()} MAP相関分析</h3>
                    {correlation_matrix}
                    {strong_correlations}
                    """)
            
            return f"""
            <div class="section" id="correlations">
                <h2>4. 相関分析</h2>
                <p>各指標間の相関関係を分析します。相関係数の絶対値が0.7以上を強い相関、
                0.4-0.7を中程度の相関、0.4未満を弱い相関として解釈します。</p>
                {''.join(correlation_sections)}
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"相関分析構築エラー: {e}")
            return '<div class="section"><h2>4. 相関分析</h2><p>エラーが発生しました。</p></div>'
    
    def _build_conclusions(self, results: Dict[str, Dict]) -> str:
        """結論・提言の構築"""
        try:
            conclusions = []
            recommendations = []
            
            # 結果に基づく結論生成
            for analysis_type, data in results.items():
                metrics = data.get('metrics', {})
                
                # Integration分析
                if 'integration' in metrics:
                    integ_series = metrics['integration']
                    integ_mean = integ_series.mean()
                    integ_std = integ_series.std()
                    
                    if integ_std / integ_mean > 0.5:  # 変動係数が大きい場合
                        conclusions.append(f"{analysis_type.upper()} MAPにおいて、Integration値のばらつきが大きく、空間の統合性に格差があることが示されました。")
                        recommendations.append("統合性の低い地域への交通インフラ整備や歩行環境改善を検討することを推奨します。")
                
                # Connectivity分析
                if 'connectivity' in metrics:
                    conn_series = metrics['connectivity']
                    high_conn_count = len(conn_series[conn_series > conn_series.quantile(0.9)])
                    
                    conclusions.append(f"{analysis_type.upper()} MAPにおいて、{high_conn_count}箇所が高接続性を示し、重要な交通結節点として機能しています。")
                    recommendations.append("高接続性地点の交通容量や安全性の向上を優先的に検討することを推奨します。")
                
                # Intelligibility分析
                if 'intelligibility' in metrics:
                    intell_value = metrics['intelligibility']
                    if isinstance(intell_value, (int, float)):
                        if intell_value > 0.6:
                            conclusions.append(f"{analysis_type.upper()} MAPのIntelligibility値が{intell_value:.3f}と高く、空間認知性が良好です。")
                        elif intell_value < 0.3:
                            conclusions.append(f"{analysis_type.upper()} MAPのIntelligibility値が{intell_value:.3f}と低く、空間認知性に課題があります。")
                            recommendations.append("ランドマークの設置や視線誘導サインの充実により、空間認知性の向上を図ることを推奨します。")
            
            return f"""
            <div class="section" id="conclusions">
                <h2>5. 結論・提言</h2>
                
                <h3>主要な発見</h3>
                <ul>
                    {''.join([f'<li>{conclusion}</li>' for conclusion in conclusions])}
                </ul>
                
                <h3>改善提言</h3>
                <ul>
                    {''.join([f'<li>{recommendation}</li>' for recommendation in recommendations])}
                </ul>
                
                <div class="highlight">
                    <h4>今後の展開</h4>
                    <p>本分析結果を基に、以下の追加調査・検討を推奨します：</p>
                    <ul>
                        <li>実地調査による分析結果の検証</li>
                        <li>時間帯別・季節別の変動分析</li>
                        <li>他地域との比較分析</li>
                        <li>将来計画の影響予測</li>
                    </ul>
                </div>
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"結論・提言構築エラー: {e}")
            return '<div class="section"><h2>5. 結論・提言</h2><p>エラーが発生しました。</p></div>'
    
    def _build_appendix(self) -> str:
        """付録の構築"""
        try:
            return """
            <div class="section" id="appendix">
                <h2>6. 付録</h2>
                
                <h3>Space Syntax理論について</h3>
                <p>Space Syntaxは、ビル・ヒリアー（Bill Hillier）らによって開発された都市・建築空間の分析理論です。
                空間の形態的特性を定量的に分析し、人間の行動や社会活動との関係を明らかにします。</p>
                
                <h3>主要指標の定義</h3>
                <div class="info-grid">
                    <div class="info-card">
                        <h4>Integration（統合性）</h4>
                        <p>ある空間が全体のシステムからどの程度アクセスしやすいかを示す指標。
                        値が高いほど中心性が高く、多くの人が訪れやすい空間である。</p>
                    </div>
                    <div class="info-card">
                        <h4>Connectivity（接続性）</h4>
                        <p>ある空間に直接接続している空間の数。
                        値が高いほど多くの選択肢があり、交通の結節点としての重要性が高い。</p>
                    </div>
                    <div class="info-card">
                        <h4>Choice（選択性）</h4>
                        <p>ある空間が他の空間間の最短経路上にある頻度。
                        値が高いほど通過交通が多く、商業活動に適した空間である。</p>
                    </div>
                    <div class="info-card">
                        <h4>Intelligibility（理解容易性）</h4>
                        <p>ConnectivityとIntegrationの相関係数。
                        値が高いほど局所的な情報から全体的な構造を理解しやすい。</p>
                    </div>
                </div>
                
                <h3>解析手法</h3>
                <ul>
                    <li><strong>Axial Map:</strong> 道路中心線を統合した軸線による解析</li>
                    <li><strong>Segment Map:</strong> 道路セグメント単位での角度重み付き解析</li>
                    <li><strong>Local Analysis:</strong> 限定された半径内での局所的解析</li>
                    <li><strong>Global Analysis:</strong> システム全体を対象とした大域的解析</li>
                </ul>
                
                <h3>参考文献</h3>
                <ul>
                    <li>Hillier, B., & Hanson, J. (1984). The Social Logic of Space. Cambridge University Press.</li>
                    <li>Hillier, B. (2007). Space is the Machine. Space Syntax.</li>
                    <li>Al-Sayed, K., Turner, A., Hillier, B., Iida, S., & Penn, A. (2014). Space Syntax Methodology. Bartlett School of Architecture, UCL.</li>
                </ul>
            </div>
            </div>
            """
            
        except Exception as e:
            self.logger.error(f"付録構築エラー: {e}")
            return '<div class="section"><h2>6. 付録</h2><p>エラーが発生しました。</p></div>'
    
    def _build_correlation_matrix_html(self, correlations: pd.DataFrame) -> str:
        """相関行列HTMLの構築"""
        try:
            rows = []
            
            # ヘッダー行
            header_cells = ['<td></td>']  # 左上は空
            for col in correlations.columns:
                header_cells.append(f'<th>{self._get_metric_label(col)}</th>')
            rows.append(f"<tr>{''.join(header_cells)}</tr>")
            
            # データ行
            for idx, row in correlations.iterrows():
                cells = [f'<th>{self._get_metric_label(idx)}</th>']
                for col in correlations.columns:
                    value = row[col]
                    if pd.isna(value):
                        cells.append('<td>-</td>')
                    else:
                        css_class = self._get_correlation_css_class(abs(value))
                        cells.append(f'<td class="{css_class}">{value:.3f}</td>')
                rows.append(f"<tr>{''.join(cells)}</tr>")
            
            return f"""
            <table class="metrics-table">
                <thead>
                    <tr><th colspan="{len(correlations.columns)+1}">相関係数行列</th></tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            """
            
        except Exception as e:
            self.logger.error(f"相関行列HTML構築エラー: {e}")
            return "<p>相関行列の表示でエラーが発生しました。</p>"
    
    def _analyze_strong_correlations(self, correlations: pd.DataFrame) -> str:
        """強い相関の分析"""
        try:
            strong_pairs = []
            
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    col1, col2 = correlations.columns[i], correlations.columns[j]
                    corr_value = correlations.iloc[i, j]
                    
                    if pd.isna(corr_value):
                        continue
                    
                    abs_corr = abs(corr_value)
                    if abs_corr >= 0.7:
                        strength = "強い正の相関" if corr_value > 0 else "強い負の相関"
                        strong_pairs.append(f"""
                        <li><strong>{self._get_metric_label(col1)}</strong> と 
                        <strong>{self._get_metric_label(col2)}</strong>: 
                        {corr_value:.3f} ({strength})</li>
                        """)
            
            if strong_pairs:
                return f"""
                <div class="highlight">
                    <h4>注目すべき相関関係</h4>
                    <ul>
                        {''.join(strong_pairs)}
                    </ul>
                </div>
                """
            else:
                return "<p>強い相関関係（|r| ≥ 0.7）は検出されませんでした。</p>"
                
        except Exception as e:
            self.logger.error(f"強い相関分析エラー: {e}")
            return "<p>相関分析でエラーが発生しました。</p>"
    
    def _convert_html_to_pdf(self, html_path: Path, pdf_path: Path):
        """HTMLからPDFへの変換"""
        try:
            # weasyprint使用（利用可能な場合）
            try:
                import weasyprint
                weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
                self.logger.info("weasyprint でPDF変換完了")
                return
            except ImportError:
                pass
            
            # pdfkit使用（利用可能な場合）
            try:
                import pdfkit
                pdfkit.from_file(str(html_path), str(pdf_path))
                self.logger.info("pdfkit でPDF変換完了")
                return
            except ImportError:
                pass
            
            # フォールバック: HTMLファイルをそのまま保存
            self.logger.warning("PDF変換ライブラリが利用できません。HTMLファイルを保存しました。")
            
        except Exception as e:
            self.logger.error(f"PDF変換エラー: {e}")
    
    def _get_metric_label(self, metric_name: str) -> str:
        """指標名の日本語ラベル取得"""
        label_map = {
            'connectivity': 'Connectivity（接続性）',
            'integration': 'Integration（統合性）',
            'choice': 'Choice（選択性）',
            'depth': 'Depth（深度）',
            'intelligibility': 'Intelligibility（理解容易性）',
            'angular_integration': 'Angular Integration（角度統合性）',
            'angular_choice': 'Angular Choice（角度選択性）',
            'angular_depth': 'Angular Depth（角度深度）'
        }
        
        for base_name, label in label_map.items():
            if metric_name.startswith(base_name):
                if '_r' in metric_name:
                    radius = metric_name.split('_r')[-1]
                    return f"{label} (R{radius})"
                return label
        
        return metric_name
    
    def _get_correlation_css_class(self, abs_corr: float) -> str:
        """相関の強さに応じたCSSクラス取得"""
        if abs_corr >= 0.7:
            return "correlation-strong"
        elif abs_corr >= 0.4:
            return "correlation-moderate"
        else:
            return "correlation-weak"
    
    def _interpret_intelligibility(self, value: float) -> str:
        """Intelligibility値の解釈"""
        if value >= 0.7:
            return "非常に高い理解容易性。局所的な情報から全体構造を把握しやすい環境です。"
        elif value >= 0.5:
            return "高い理解容易性。比較的分かりやすい空間構造です。"
        elif value >= 0.3:
            return "中程度の理解容易性。一部で方向感覚を失いやすい可能性があります。"
        else:
            return "低い理解容易性。複雑で理解しにくい空間構造です。"
    
    def _get_metric_descriptions(self) -> Dict[str, str]:
        """指標説明の取得"""
        return {
            'connectivity': '各空間に直接接続している空間の数を示します。値が高いほど多くの選択肢がある交通結節点です。',
            'integration': '空間がシステム全体からどの程度アクセスしやすいかを示します。値が高いほど中心性が高く、人が集まりやすい場所です。',
            'choice': '空間が他の空間間の最短経路上にある頻度を示します。値が高いほど通過交通が多く、商業活動に適しています。',
            'depth': 'システム内の他の全ての空間までの平均距離を示します。値が小さいほど中心的な位置にあります。',
            'angular': '角度変化を考慮した分析で、実際の移動パターンをより正確に反映します。'
        }
    
    def _get_analysis_description(self, analysis_type: str) -> str:
        """解析タイプの説明"""
        descriptions = {
            'axial': 'Axial Map解析では、道路中心線を統合した軸線単位で空間特性を分析します。都市レベルでの大局的な構造把握に適しています。',
            'segment': 'Segment Map解析では、道路セグメント単位で角度重み付きの分析を行います。実際の歩行・移動パターンをより詳細に反映します。'
        }
        return descriptions.get(analysis_type, f'{analysis_type}解析の詳細な説明。')