#!/usr/bin/env bash
# Calvano et al. (2020) 再現実験 一括実行スクリプト

echo "🎯 Calvano et al. (2020) 再現実験開始"
echo "==================================="

# 結果ディレクトリ作成
mkdir -p results figs

# Table A.2 再現
echo "📊 Table A.2 再現実行中..."
python scripts/table_a2.py --episodes 1000 --n-seeds 5 \
       --csv results/table_a2_final.csv --json results/table_a2_final.json

# Figure 2b 再現
echo "📈 Figure 2b 再現実行中..."
python scripts/plot_figure2b.py --episodes 1000 --out figs/figure2b.png

echo "✅ 全実験完了！"
echo "結果ファイル:"
echo "  - results/table_a2_final.csv"
echo "  - results/table_a2_final.json"
echo "  - figs/figure2b.png"
