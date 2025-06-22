"""
Integration tests for Calvano et al. (2020) implementation.
Tests the complete pipeline from training to analysis.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_agents
from env import DemandEnvironment
from agent import QLearningAgent
from grid import make_grid


class TestIntegration:
    """Integration tests for complete Calvano pipeline."""
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline with small parameters."""
        # Run short training
        agents, env, history = train_agents(
            n_episodes=5,
            iterations_per_episode=100,
            rng_seed=42,
            verbose=False
        )
        
        # Check basic functionality
        assert len(agents) == 2
        assert isinstance(env, DemandEnvironment)
        assert 'training_summary' in history
        assert 'individual_profits' in history
        assert 'joint_profits' in history
        
        # Check profit values are reasonable
        final_profit = history['training_summary']['individual_profit']
        assert 0.0 <= final_profit <= 1.0
    
    def test_environment_equilibrium_calculation(self):
        """Test environment equilibrium calculations."""
        env = DemandEnvironment()
        
        # Test Nash equilibrium
        nash_eq = env.get_nash_equilibrium()
        assert 1.0 < nash_eq['prices'][0] < 2.0  # Reasonable range
        
        # Test cooperative outcome
        coop_eq = env.get_collusive_outcome()
        assert coop_eq['prices'][0] > nash_eq['prices'][0]  # Cooperative should be higher
        
        # Test demand calculation
        prices = [nash_eq['prices'][0], nash_eq['prices'][0]]
        demands = env.compute_demands(prices)
        assert len(demands) == 2
        assert all(0 < d < 1 for d in demands)  # Reasonable demand range
    
    def test_agent_initialization(self):
        """Test agent initialization with dynamic grid."""
        env = DemandEnvironment()
        agent = QLearningAgent(0, env, rng_seed=42)
        
        # Check initialization
        assert agent.agent_id == 0
        assert len(agent.price_grid) == 15
        assert agent.q_table.shape[1] == 15  # 15 states, 15 actions for k=1
        assert agent.current_epsilon == 1.0
        assert agent.beta == 9.21e-5
    
    def test_price_grid_generation(self):
        """Test dynamic price grid generation."""
        p_nash = 1.125
        p_coop = 1.25
        xi = 0.1
        m = 15
        
        grid = make_grid(p_nash, p_coop, xi, m)
        
        # Check grid properties
        assert len(grid) == m
        assert grid[0] < p_nash  # First price below Nash
        assert grid[-1] > p_coop  # Last price above cooperative
        
        # Check spacing
        spacings = [grid[i+1] - grid[i] for i in range(len(grid)-1)]
        assert all(abs(s - spacings[0]) < 1e-10 for s in spacings)  # Equal spacing
    
    def test_specification_compliance(self):
        """Test compliance with Calvano et al. (2020) specifications."""
        env = DemandEnvironment()
        
        # Check environment parameters
        assert env.a_0 == 0.0  # Outside option
        assert env.mu == 0.25  # Price sensitivity
        assert env.c == 1.0    # Marginal cost
        assert env.a_i == 2.0  # Product quality
        
        # Check agent parameters
        agent = QLearningAgent(0, env)
        assert agent.alpha == 0.15     # Learning rate
        assert agent.gamma == 0.95     # Discount factor
        assert agent.beta == 9.21e-5   # Epsilon decay
        assert agent.memory_length == 1  # k=1 periods
        assert len(agent.price_grid) == 15  # m=15 points
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seeds."""
        # Run same experiment twice
        results1 = train_agents(
            n_episodes=3,
            iterations_per_episode=50,
            rng_seed=999,
            verbose=False
        )
        
        results2 = train_agents(
            n_episodes=3,
            iterations_per_episode=50,
            rng_seed=999,
            verbose=False
        )
        
        # Check reproducibility
        profit1 = results1[2]['training_summary']['individual_profit']
        profit2 = results2[2]['training_summary']['individual_profit']
        
        assert abs(profit1 - profit2) < 1e-10  # Should be identical


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
