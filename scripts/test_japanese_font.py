# test_japanese_font.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
    print("✅ japanize_matplotlib インポート成功")
    
    # テスト用グラフ
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.title('日本語テスト - グラフタイトル')
    plt.xlabel('横軸ラベル')
    plt.ylabel('縦軸ラベル')
    plt.savefig('japanese_font_test.png', dpi=150, bbox_inches='tight')
    print("✅ 日本語フォントテスト完了: japanese_font_test.png")
    
except ImportError as e:
    print(f"❌ japanize_matplotlib インポートエラー: {e}")
    print("pip install japanize-matplotlib でインストールしてください")