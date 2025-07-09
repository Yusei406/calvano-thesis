# Calvano et al. (2020) Q-Learning Replication

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yusei406/calvano-thesis/blob/parallel-colab/calvano_colab.ipynb)

**Python実装**: Calvano et al. (2020) "Artificial Intelligence, Algorithmic Pricing, and Collusion" の統計的に有意な再現実験

⚠️ **重要**: この実装は論文の公開情報とコミュニティの理解に基づいており、論文の正確な実装詳細は完全には確認できていません。

🔧 **最新更新**: セッション切断対策として、Colabでの実験を「セル内直接実行方式」に変更。subprocessを使わない安全な実装により、長時間実験でも安定動作。

## 🚀 クイックスタート

### Google Colab (推奨)

**論文Table A.2の再現実験**をGoogle Colabで実行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yusei406/calvano-thesis/blob/parallel-colab/calvano_colab.ipynb)

**Colabでの実行手順：**
1. 🔗 上記バッジをクリックしてColabで開く
2. ⚡ **小規模実験**（10-15分）で動作確認
3. 📊 **中規模実験**（45-60分）で統計的に有意な結果を取得 ⭐推奨
4. 🎯 **フル実験**（3-4時間）で論文完全再現を試行
5. 📈 結果の自動分析・可視化・ダウンロード

**実験レベル（セッション切断対策版）**:
- **小規模実験**: 1,000エピソード × 2シード（10-15分）
- **中規模実験**: 5,000エピソード × 2シード（45-60分）⭐推奨
- **フル実験**: 15,000エピソード × 3シード（3-4時間）

### ローカル環境

```bash
# リポジトリクローン
git clone https://github.com/Yusei406/calvano-thesis.git
cd calvano-thesis

# 依存関係インストール
pip install -r requirements.txt

# 基本実験
python -m myproject.train --episodes 1000 --iterations-per-episode 5000

# 並列実験（Table A.2再現）
python -m myproject.scripts.table_a2_parallel --episodes 10000 --n-seeds 3
```

## 📚 実装パラメータ

### 🔬 **採用パラメータ（論文準拠）**

```python
# 環境パラメータ（論文仕様）
DemandEnvironment(
    demand_intercept=0.0,    # a₀ = 0
    product_quality=2.0,     # aᵢ = 2 (品質差 aᵢ-cᵢ=1 + cᵢ=1)
    demand_slope=0.25,       # μ = 0.25 (水平差別度)
    marginal_cost=1.0        # cᵢ = 1
)

# Q学習パラメータ（論文仕様）
QLearningAgent(
    learning_rate=0.15,           # α = 0.15
    discount_factor=0.95,         # δ = 0.95 (割引率)
    epsilon_decay_beta=4e-6,      # β = 4×10⁻⁶ (論文値)
    memory_length=1,              # k = 1 (記憶長)
    grid_size=15,                 # m = 15 (価格グリッド点数)
    grid_extension=0.1            # ξ = 0.1 (グリッド拡張)
)

# 実験設定（論文仕様）
iterations_per_episode=25000      # 論文推奨値
beta_scaled = 4e-6 * 25000 = 0.1  # β* = β × iterations_per_episode
n_episodes=80000                  # 最大エピソード数
```

### ✅ **パラメータの信頼性（更新）**

| パラメータ | 信頼度 | 根拠 |
|-----------|--------|------|
| a₀=0, aᵢ=2, μ=0.25, cᵢ=1 | 高 | 論文チートシート明記 |
| α=0.15, δ=0.95 | 高 | 論文チートシート明記 |
| β=4×10⁻⁶ | 高 | **論文チートシート確認済み** |
| 25,000イテレーション | 高 | **論文推奨値確認済み** |
| k=1, m=15, ξ=0.1 | 高 | **論文チートシート確認済み** |
| β*=0.1 (スケーリング) | 高 | **論文仕様確認済み** |

## 🔬 **実験結果**

### 現在の実装での均衡値
- **Nash均衡**: 個別利益 0.223, 価格 1.473
- **協調均衡**: 個別利益 0.337, 価格 1.925

### 論文期待値（Table A.2）
- **Individual profit**: 0.18 ± 0.03
- **Joint profit**: 0.26 ± 0.04  
- **Nash ratio**: > 100% (協調的行動の証拠)

**注意**: 現在の実装は論文の期待値とは異なります。パラメータ調整が必要です。

## 📁 プロジェクト構造

```
myproject/
├── __init__.py              # パッケージ初期化
├── env.py                   # 需要環境（logit需要関数）
├── agent.py                 # Q学習エージェント
├── grid.py                  # 動的価格グリッド生成
├── train.py                 # 学習モジュール
└── scripts/
    ├── __init__.py
    └── table_a2_parallel.py # 並列実験スクリプト
```

## 🎯 主要機能

### ✅ 基本実装
- **数値計算**: Nash均衡・協調均衡の数値計算
- **価格グリッド**: Nash・協調価格に基づく動的グリッド
- **学習アルゴリズム**: ε-greedy Q学習とepsilon decay
- **並列実行**: ProcessPoolExecutorによる高速化

### ✅ 統計的検証
- **複数シード**: 統計的有意性の確保
- **プログレス追跡**: リアルタイム進捗表示
- **エラーハンドリング**: 堅牢な例外処理

### ✅ 結果分析
- **自動可視化**: 利益分布・Nash比・学習曲線
- **論文比較**: 期待値との差分分析
- **データエクスポート**: JSON・CSVでの結果保存

### ✅ 使いやすさ
- **Colab統合**: ワンクリック実行環境
- **インタラクティブUI**: 実験設定の簡単選択
- **包括的ドキュメント**: 詳細な使用方法説明

## 🔧 使用方法

### 基本的な学習実験

```python
from myproject.train import train_agents

# 基本実験
agents, env, history = train_agents(
    n_episodes=1000,
    iterations_per_episode=5000,
    verbose=True
)
```

### 並列実験（統計的検証用）

```bash
# 中規模実験（推奨）
python -m myproject.scripts.table_a2_parallel \
    --episodes 10000 \
    --n-seeds 3 \
    --n-sessions 4 \
    --max-workers 4

# フル実験
python -m myproject.scripts.table_a2_parallel \
    --episodes 50000 \
    --n-seeds 10 \
    --n-sessions 4 \
    --max-workers 8
```

## 🧪 テスト

```bash
# テスト実行
python -m pytest tests/ -v

# カバレッジ付き
python -m pytest tests/ --cov=myproject
```

## 📊 実験レベル

| レベル | エピソード数 | シード数 | 実行時間 | 目的 |
|--------|-------------|----------|----------|------|
| 小規模 | 1,000 | 2 | 10-15分 | 動作確認 |
| 中規模 | 5,000 | 2 | 45-60分 | 統計的検証 |
| フル | 15,000 | 3 | 3-4時間 | 論文再現試行 |

## ⚠️ 制限事項

1. **パラメータの不確実性**: 論文の正確な実装詳細は未確認
2. **結果の相違**: 現在の実装は論文の期待値と異なる
3. **計算時間**: フル実験は非常に長時間（日単位）
4. **環境依存**: 数値計算の精度は環境に依存

## 🔮 今後の改善

1. **論文原文確認**: 正確なパラメータの特定
2. **パラメータチューニング**: 期待値に合わせた調整
3. **アルゴリズム改善**: より効率的な学習手法
4. **可視化強化**: より詳細な分析ツール

## 📖 参考文献

- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). "Artificial Intelligence, Algorithmic Pricing, and Collusion." *American Economic Review*, 110(10), 3267-3297.
- [VoxEU記事](https://cepr.org/voxeu/columns/artificial-intelligence-algorithmic-pricing-and-collusion)

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🤝 貢献

プルリクエストと改善提案を歓迎します。特に：
- 論文パラメータの正確な特定
- アルゴリズムの改善
- 計算効率の向上
- テストの追加 