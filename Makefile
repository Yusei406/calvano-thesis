# Calvano et al. (2020) Q-Learning Replication Makefile

.PHONY: quick mid full test clean help

help:
	@echo "🎯 Calvano Q-Learning 実験コマンド"
	@echo "=================================="
	@echo "make quick    - 開発用短縮ラン (100ep×2seed, ~5分)"
	@echo "make mid      - 卒論用中規模ラン (5000ep×5seed, ~1時間)"
	@echo "make full     - 論文規模フルラン (50000ep×10seed, ~36時間)"
	@echo "make test     - 数学的検証テスト"
	@echo "make clean    - 結果ファイル削除"

quick:
	@echo "🚀 開発用短縮ラン実行中..."
	python scripts/table_a2.py --episodes 100 --n-seeds 2 \
		--csv results/table_a2_quick.csv --json results/table_a2_quick.json
	python scripts/plot_figure2b.py --episodes 100 --out figs/figure2b_quick.png --fast

mid:
	@echo "📊 卒論用中規模ラン実行中..."
	python scripts/table_a2.py --episodes 5000 --n-seeds 5 \
		--csv results/table_a2_mid.csv --json results/table_a2_mid.json
	python scripts/plot_figure2b.py --episodes 5000 --out figs/figure2b_mid.png

full:
	@echo "🎯 論文規模フルラン実行中..."
	python scripts/table_a2.py --episodes 50000 --n-seeds 10 \
		--csv results/table_a2_full.csv --json results/table_a2_full.json
	python scripts/plot_figure2b.py --episodes 50000 --out figs/figure2b_full.png

test:
	@echo "�� 数学的検証テスト実行中..."
	python tests/test_math.py
	python tests/test_integration.py

clean:
	@echo "🧹 結果ファイル削除中..."
	rm -f results/table_a2_*.csv results/table_a2_*.json
	rm -f figs/figure2b_*.png
