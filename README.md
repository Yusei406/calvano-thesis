# Calvano et al. (2020) - Strict Paper Replication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-Calvano%20et%20al.%20(2020)-blue)](https://www.aeaweb.org/articles?id=10.1257/aer.20190623)

**100% specification-compliant replication of Calvano et al. (2020) "Artificial Intelligence, Algorithmic Pricing, and Collusion"**

## 🎯 7 Key Corrections Implemented

1. **✅ Dynamic Price Grid**: 15-point grid generated from Nash/Cooperative prices  
2. **✅ Exact Demand Parameters**: a₀=0, μ=0.25, c=1, a_i=2
3. **✅ Correct Demand Formula**: Logit with proper outside option handling
4. **✅ Memory Length**: k=1 (single period opponent prices)
5. **✅ Episode Structure**: 25,000 iterations per episode
6. **✅ Epsilon Decay**: ε(t) = exp(-βt), β=4×10⁻⁶ on iteration scale
7. **✅ Price Units**: Currency units (not normalized)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/[USER]/calvano-strict.git
cd calvano-strict
pip install -r requirements.txt
```

### Run Replication
```bash
# Quick test (2 minutes)
python scripts/table_a2_simple.py --episodes 5 --n-seeds 2

# Full Table A.2 replication (24+ hours)
python scripts/run_reproduction.py --episodes 50000 --n-seeds 10

# Generate Figure 2b
python scripts/plot_figure2b.py --fast
```

### Docker
```bash
docker build -t calvano-strict .
docker run calvano-strict
```

## 📊 Implementation Details

### Demand Model (Exact Calvano)
```python
# Parameters
a₀ = 0.0        # Outside option quality
a_i = 2.0       # Product quality  
μ = 0.25        # Price sensitivity
c = 1.0         # Marginal cost

# Demand function
D_i = exp((a_i - p_i) / μ) / (exp(a₀/μ) + Σ exp((a_j - p_j)/μ))
```

### Q-Learning (Paper Specification)
- **Learning Rate**: α = 0.15
- **Discount Factor**: γ = 0.95
- **Epsilon Decay**: ε(t) = exp(-4×10⁻⁶ × t)
- **State Space**: k=1 opponent price memory
- **Action Space**: 15-point dynamic grid

## 📚 Citation

```bibtex
@article{calvano2020artificial,
  title={Artificial Intelligence, Algorithmic Pricing, and Collusion},
  author={Calvano, Emilio and Calzolari, Giacomo and Denicol{\`o}, Vincenzo and Pastorello, Sergio},
  journal={American Economic Review},
  volume={110},
  number={10},
  pages={3267--3297},
  year={2020},
  publisher={American Economic Association}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Status**: ✅ All 7 corrections implemented | ✅ Paper specification 100% compliant
