#!/bin/bash
# run_station_analysis.sh
# 駅周辺ネットワーク分析実行スクリプト

echo "🚉 駅周辺道路ネットワーク分析デモ実行スクリプト"
echo "=================================================="

# 仮想環境とパッケージの確認
echo "📋 環境確認中..."

# 必要なディレクトリを作成
mkdir -p data
mkdir -p station_analysis_output

# 設定ファイルの存在確認
if [ ! -f "station_config.json" ]; then
    echo "⚠️ 設定ファイル (station_config.json) が見つかりません"
    echo "   サンプル設定ファイルを作成しますか？ (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "📝 サンプル設定ファイル作成中..."
        # サンプル設定ファイルをPythonスクリプトで生成
        python3 -c "
import json
sample_config = {
    'analysis_settings': {
        'radius_meters': 800,
        'network_type': 'drive',
        'include_analysis': ['basic', 'axial', 'integration'],
        'save_graphml': True,
        'save_visualization': True,
        'background_map': True
    },
    'output_directory': 'station_analysis_output',
    'stations': [
        {
            'id': 'shinjuku',
            'name': '新宿駅',
            'location': 'Shinjuku Station, Tokyo, Japan',
            'coordinates': None,
            'graphml_path': None,
            'description': '日本最大のターミナル駅'
        },
        {
            'id': 'shibuya', 
            'name': '渋谷駅',
            'location': 'Shibuya Station, Tokyo, Japan',
            'coordinates': [35.6580, 139.7016],
            'graphml_path': None,
            'description': '若者文化の中心地'
        }
    ]
}
with open('station_config.json', 'w', encoding='utf-8') as f:
    json.dump(sample_config, f, ensure_ascii=False, indent=2)
print('✅ サンプル設定ファイルを作成しました')
"
        echo "📝 station_config.json を編集してから再実行してください"
        exit 0
    else
        echo "❌ 設定ファイルが必要です"
        exit 1
    fi
fi

echo "✅ 設定ファイル確認完了"

# 依存関係の確認
echo "🔍 依存関係確認中..."

# Pythonスクリプトで依存関係をチェック
python3 -c "
import sys
required_packages = ['osmnx', 'networkx', 'pandas', 'matplotlib', 'numpy', 'geopandas', 'contextily']
missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f'   ✅ {package}')
    except ImportError:
        print(f'   ❌ {package}')
        missing.append(package)

if missing:
    print(f'\\n⚠️ 不足パッケージ: {missing}')
    print('インストール: uv add ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✅ 全ての依存関係が満たされています')
" || {
    echo "❌ 依存関係に問題があります"
    exit 1
}

# 分析実行の確認
echo ""
echo "📋 分析設定:"
python3 -c "
import json
with open('station_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
settings = config.get('analysis_settings', {})
stations = config.get('stations', [])
print(f'   分析半径: {settings.get(\"radius_meters\", 800)}m')
print(f'   ネットワーク種別: {settings.get(\"network_type\", \"drive\")}')
print(f'   対象駅数: {len(stations)}駅')
print(f'   出力先: {config.get(\"output_directory\", \"station_analysis_output\")}')
"

echo ""
echo "🚀 分析を実行しますか？ (y/n)"
read -r execute
if [ "$execute" != "y" ] && [ "$execute" != "Y" ]; then
    echo "👋 実行をキャンセルしました"
    exit 0
fi

# 分析実行
echo ""
echo "🚉 駅周辺ネットワーク分析開始..."
echo "=================================================="

# uvを使用してPythonスクリプトを実行
if command -v uv &> /dev/null; then
    echo "📦 uv を使用して実行中..."
    uv run python examples/station_analysis_demo.py
else
    echo "🐍 Python を直接使用して実行中..."
    python3 examples/station_analysis_demo.py
fi

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 分析完了!"
    echo "📁 結果ファイル:"
    if [ -d "station_analysis_output" ]; then
        ls -la station_analysis_output/
    fi
    echo ""
    echo "💡 生成されたファイル:"
    echo "   - *.png: 各駅のネットワーク図・軸線分析図"
    echo "   - *.graphml: ネットワークデータ"
    echo "   - station_comparison.csv: 駅間比較データ"
    echo "   - station_comparison_charts.png: 比較チャート"
    echo "   - station_analysis_report_*.md: 統合レポート"
else
    echo "❌ 分析中にエラーが発生しました"
    exit 1
fi