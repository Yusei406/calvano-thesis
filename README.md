# Calvano et al. (2020) Q-Learning Replication

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yusei406/calvano-thesis/blob/parallel-colab/calvano_colab.ipynb)

Complete Python implementation of Calvano et al. (2020) "Artificial Intelligence, Algorithmic Pricing, and Collusion" with modern package structure and parallel execution support.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Yusei406/calvano-thesis.git
cd calvano-thesis

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Single training run:**
```bash
python -m myproject.train --episodes 1000 --verbose
```

**Parallel experiment:**
```bash
python -m myproject.scripts.table_a2_parallel --episodes 50000 --n-seeds 10 --n-sessions 4
```

**Programmatic usage:**
```python
import myproject
agents, env, history = myproject.train_agents(n_episodes=1000)
print(f"Final profit: {history['training_summary']['individual_profit']:.4f}")
```

## 📁 Project Structure

```
myproject/
├── __init__.py              # Package initialization
├── env.py                   # Demand environment
├── agent.py                 # Q-learning agent
├── grid.py                  # Price grid generation
├── train.py                 # Training module
└── scripts/
    ├── __init__.py
    └── table_a2_parallel.py # Parallel execution script
```

## 🎯 Key Features

### ✅ Complete Calvano Specification
- **25,000 iterations per episode** (Table A.2 requirement)
- **β = 4×10⁻⁶** epsilon decay parameter
- **15-point price grid** with ξ=0.1 extension
- **Memory length k=1** for state encoding
- **Dynamic grid** based on Nash/Cooperative equilibria

### ✅ Modern Package Structure
- **Relative imports** maintained throughout
- **Module execution** support (`python -m myproject.train`)
- **Package import** support (`import myproject`)
- **No path manipulation** required

### ✅ Parallel Execution
- **Multi-session per seed** support
- **ProcessPoolExecutor** for CPU utilization
- **Progress tracking** and error handling
- **JSON result export** with timestamps

### ✅ Input Validation
- **Parameter bounds** checking
- **Type validation** for all inputs
- **Meaningful error messages**

### ✅ Beta Normalization
- **Automatic β_effective** calculation
- **Convergence tracking** with epsilon values
- **Training history** with beta information

## 📊 Table A.2 Replication

### Target Results
- **Individual profit**: 0.18 ± 0.03
- **Joint profit**: 0.26 ± 0.04
- **Required**: 50,000 episodes × 10 seeds

### Execution Commands

**Full replication (3+ hours):**
```bash
python -m myproject.scripts.table_a2_parallel \
  --episodes 50000 \
  --n-seeds 10 \
  --n-sessions 4 \
  --max-workers 4
```

**Quick test (5 minutes):**
```bash
python -m myproject.scripts.table_a2_parallel \
  --episodes 1000 \
  --n-seeds 2 \
  --n-sessions 2
```

## 🔧 Configuration

### Environment Parameters (Calvano et al. 2020)
- **a₀ = 0** (demand intercept)
- **aᵢ = 2** (product quality)
- **μ = 0.25** (demand slope)
- **c = 1** (marginal cost)

### Q-Learning Parameters
- **α = 0.15** (learning rate)
- **γ = 0.95** (discount factor)
- **β = 4×10⁻⁶** (epsilon decay)
- **k = 1** (memory length)

## 📈 Results Format

### Training History
```python
{
    'episodes': [...],
    'individual_profits': [...],
    'joint_profits': [...],
    'epsilon_values': [...],
    'total_iterations': 1250000,
    'beta_info': {
        'beta_raw': 4e-6,
        'iterations_per_episode': 25000,
        'beta_effective': 1.6e-10,
        'epsilon_at_convergence': 0.6065
    },
    'training_summary': {
        'final_individual_profit': 0.2710,
        'final_joint_profit': 0.5590,
        'nash_ratio_individual': 1.216,
        'final_epsilon': 0.6065
    }
}
```

### Parallel Experiment Results
```json
{
    "experiment_info": {
        "n_seeds": 10,
        "n_sessions_per_seed": 4,
        "total_sessions": 40,
        "successful_sessions": 40,
        "execution_time_seconds": 10800
    },
    "aggregated_stats": {
        "individual_profit_mean": 0.1800,
        "individual_profit_std": 0.0300,
        "joint_profit_mean": 0.2600,
        "joint_profit_std": 0.0400
    }
}
```

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest tests/

# Specific test
python -m pytest tests/test_profit_targets.py::test_beta_normalization -v
```

### Test Coverage
- **Beta normalization** verification
- **Environment parameters** validation
- **Profit targets** for different episode counts
- **Input validation** testing

## 🐳 Docker Support

```bash
# Build image
docker build -t calvano-replication .

# Run experiment
docker run -v $(pwd)/results:/app/results calvano-replication \
  python -m myproject.scripts.table_a2_parallel --episodes 1000
```

## 📚 References

- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *American Economic Review*, 110(10), 3267-3297.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original Fortran implementation by Calvano et al. (2020)
- Python adaptation with modern best practices
- Parallel execution optimization for multi-core systems

---

**Status**: ✅ Complete implementation with all Calvano specifications
**Last Updated**: December 2024
**Python Version**: 3.8+ 