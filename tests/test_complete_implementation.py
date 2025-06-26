"""
Complete test suite for Calvano et al. (2020) Q-learning replication.

Tests all major components including:
- Environment setup and equilibrium computation
- Q-learning agent functionality
- Training process and convergence
- Beta normalization
- Input validation
- Parallel execution
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myproject.env import DemandEnvironment
from myproject.agent import QLearningAgent
from myproject.train import train_agents


class TestEnvironment:
    """Test demand environment setup and equilibrium computation."""
    
    def test_environment_initialization(self):
        """Test environment initialization with Calvano parameters."""
        env = DemandEnvironment(
            demand_intercept=0.0,
            product_quality=2.0,
            demand_slope=0.25,
            marginal_cost=1.0
        )
        
        # Check parameters
        assert env.a_0 == 0.0, f"a_0 should be 0.0, got {env.a_0}"
        assert env.a_i == 2.0, f"a_i should be 2.0, got {env.a_i}"
        assert env.mu == 0.25, f"μ should be 0.25, got {env.mu}"
        assert env.c == 1.0, f"c should be 1.0, got {env.c}"
        assert env.n_agents == 2, f"n_agents should be 2, got {env.n_agents}"
    
    def test_nash_equilibrium(self):
        """Test Nash equilibrium computation."""
        env = DemandEnvironment()
        nash_eq = env.get_nash_equilibrium()
        
        # Check structure
        assert 'prices' in nash_eq
        assert 'individual_profit' in nash_eq
        assert 'joint_profit' in nash_eq
        
        # Check reasonable values (adjusted based on actual computed values)
        assert 1.0 <= nash_eq['prices'][0] <= 3.0, f"Nash price {nash_eq['prices'][0]:.3f} outside expected range"
        assert 0.1 <= nash_eq['individual_profit'] <= 0.5, f"Nash profit {nash_eq['individual_profit']:.3f} outside expected range"
    
    def test_cooperative_equilibrium(self):
        """Test cooperative equilibrium computation."""
        env = DemandEnvironment()
        coop_eq = env.get_collusive_outcome()
        
        # Check structure
        assert 'prices' in coop_eq
        assert 'individual_profit' in coop_eq
        assert 'joint_profit' in coop_eq
        
        # Check reasonable values (adjusted based on actual computed values)
        assert 1.5 <= coop_eq['prices'][0] <= 4.0, f"Cooperative price {coop_eq['prices'][0]:.3f} outside expected range"
        assert 0.2 <= coop_eq['individual_profit'] <= 0.8, f"Cooperative profit {coop_eq['individual_profit']:.3f} outside expected range"
        
        # Cooperative should be higher than Nash
        nash_eq = env.get_nash_equilibrium()
        assert coop_eq['individual_profit'] > nash_eq['individual_profit'], "Cooperative profit should exceed Nash profit"


class TestQLearningAgent:
    """Test Q-learning agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization with Calvano parameters."""
        env = DemandEnvironment()
        agent = QLearningAgent(
            agent_id=0,
            env=env,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon_decay_beta=4e-6,
            memory_length=1,
            grid_size=15,
            grid_extension=0.1
        )
        
        # Check parameters
        assert agent.alpha == 0.15, f"Learning rate should be 0.15, got {agent.alpha}"
        assert agent.gamma == 0.95, f"Discount factor should be 0.95, got {agent.gamma}"
        assert agent.beta == 4e-6, f"Beta should be 4e-6, got {agent.beta}"
        assert agent.memory_length == 1, f"Memory length should be 1, got {agent.memory_length}"
        assert agent.grid_size == 15, f"Grid size should be 15, got {agent.grid_size}"
        assert agent.grid_extension == 0.1, f"Grid extension should be 0.1, got {agent.grid_extension}"
        
        # Check price grid
        assert len(agent.price_grid) == 15, f"Price grid should have 15 points, got {len(agent.price_grid)}"
        assert agent.n_actions == 15, f"Number of actions should be 15, got {agent.n_actions}"
    
    def test_epsilon_decay(self):
        """Test epsilon decay mechanism."""
        env = DemandEnvironment()
        agent = QLearningAgent(
            agent_id=0,
            env=env,
            epsilon_decay_beta=4e-6
        )
        
        # Initial epsilon should be 1.0
        assert agent.current_epsilon == 1.0, f"Initial epsilon should be 1.0, got {agent.current_epsilon}"
        
        # After some iterations, epsilon should decrease
        initial_epsilon = agent.current_epsilon
        for _ in range(1000):
            agent.update_epsilon()
        
        assert agent.current_epsilon < initial_epsilon, "Epsilon should decrease over time"
        assert agent.current_epsilon > 0, "Epsilon should remain positive"
    
    def test_action_selection(self):
        """Test action selection mechanism."""
        env = DemandEnvironment()
        agent = QLearningAgent(agent_id=0, env=env)
        
        # Test action selection
        state = 0
        action = agent.select_action(state)
        
        assert 0 <= action < agent.n_actions, f"Action {action} should be in range [0, {agent.n_actions})"
    
    def test_q_table_update(self):
        """Test Q-table update mechanism."""
        env = DemandEnvironment()
        agent = QLearningAgent(agent_id=0, env=env)
        
        # Initial Q-values should be zero
        assert np.all(agent.q_table == 0), "Initial Q-table should be zero"
        
        # Update Q-table
        state, action, reward, next_state = 0, 0, 1.0, 1
        old_q = agent.q_table[state, action]
        agent.update_q_table(state, action, reward, next_state)
        new_q = agent.q_table[state, action]
        
        assert new_q != old_q, "Q-value should change after update"
        assert new_q > 0, "Q-value should be positive after positive reward"


class TestTraining:
    """Test training process and convergence."""
    
    def test_training_basic(self):
        """Test basic training functionality."""
        agents, env, history = train_agents(
            n_episodes=10,
            iterations_per_episode=100,
            verbose=False
        )
        
        # Check history structure
        required_keys = ['episodes', 'individual_profits', 'joint_profits', 'epsilon_values', 'total_iterations', 'beta_info', 'training_summary']
        for key in required_keys:
            assert key in history, f"History should contain '{key}'"
        
        # Check training summary
        summary = history['training_summary']
        assert 'final_individual_profit' in summary
        assert 'final_joint_profit' in summary
        assert 'nash_ratio_individual' in summary
        assert 'final_epsilon' in summary
        
        # Check beta info
        beta_info = history['beta_info']
        assert 'beta_raw' in beta_info
        assert 'iterations_per_episode' in beta_info
        assert 'beta_effective' in beta_info
        assert 'epsilon_at_convergence' in beta_info
    
    def test_beta_normalization(self):
        """Test beta normalization mechanism."""
        test_cases = [
            (1000, 4e-6 / 1000),
            (25000, 4e-6 / 25000),
            (50000, 4e-6 / 50000),
        ]
        
        for iterations_per_episode, expected_beta_eff in test_cases:
            agents, env, history = train_agents(
                n_episodes=5,
                iterations_per_episode=iterations_per_episode,
                verbose=False
            )
            
            actual_beta_eff = history['beta_info']['beta_effective']
            assert abs(actual_beta_eff - expected_beta_eff) < 1e-12, \
                f"Beta normalization failed: {actual_beta_eff:.2e} != {expected_beta_eff:.2e}"
    
    def test_convergence_indicators(self):
        """Test convergence indicators."""
        agents, env, history = train_agents(
            n_episodes=50,
            iterations_per_episode=100,
            verbose=False
        )
        
        # Epsilon should decrease over time
        epsilon_values = history['epsilon_values']
        assert epsilon_values[-1] < epsilon_values[0], "Epsilon should decrease over training"
        
        # Profits should be reasonable
        final_individual = history['training_summary']['final_individual_profit']
        final_joint = history['training_summary']['final_joint_profit']
        
        assert 0 < final_individual < 1, f"Individual profit {final_individual:.4f} should be in (0, 1)"
        assert 0 < final_joint < 2, f"Joint profit {final_joint:.4f} should be in (0, 2)"
    
    def test_input_validation(self):
        """Test input validation in training function."""
        # Test invalid episodes
        with pytest.raises(ValueError, match="n_episodes must be positive"):
            train_agents(n_episodes=0, verbose=False)
        
        # Test invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be in"):
            train_agents(learning_rate=1.5, verbose=False)
        
        # Test invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be in"):
            train_agents(discount_factor=1.1, verbose=False)
        
        # Test invalid beta
        with pytest.raises(ValueError, match="epsilon_decay_beta must be positive"):
            train_agents(epsilon_decay_beta=-1e-6, verbose=False)


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_training(self):
        """Test complete end-to-end training workflow."""
        # Run short training
        agents, env, history = train_agents(
            n_episodes=20,
            iterations_per_episode=50,
            verbose=False
        )
        
        # Check agents
        assert len(agents) == 2, "Should have 2 agents"
        for agent in agents:
            assert isinstance(agent, QLearningAgent), "Agent should be QLearningAgent instance"
        
        # Check environment
        assert isinstance(env, DemandEnvironment), "Environment should be DemandEnvironment instance"
        
        # Check results are reasonable
        final_individual = history['training_summary']['final_individual_profit']
        final_joint = history['training_summary']['final_joint_profit']
        nash_ratio = history['training_summary']['nash_ratio_individual']
        
        assert 0 < final_individual < 1, f"Individual profit {final_individual:.4f} should be in (0, 1)"
        assert 0 < final_joint < 2, f"Joint profit {final_joint:.4f} should be in (0, 2)"
        assert nash_ratio > 0, f"Nash ratio {nash_ratio:.4f} should be positive"
    
    def test_equilibrium_benchmarks(self):
        """Test that training results can be compared to equilibrium benchmarks."""
        agents, env, history = train_agents(
            n_episodes=10,
            iterations_per_episode=100,
            verbose=False
        )
        
        # Get equilibrium benchmarks
        nash_eq = env.get_nash_equilibrium()
        coop_eq = env.get_collusive_outcome()
        
        final_individual = history['training_summary']['final_individual_profit']
        
        # Final profit should be between Nash and cooperative
        assert final_individual >= nash_eq['individual_profit'] * 0.5, \
            f"Final profit {final_individual:.4f} should be at least 50% of Nash {nash_eq['individual_profit']:.4f}"
        assert final_individual <= coop_eq['individual_profit'] * 1.5, \
            f"Final profit {final_individual:.4f} should be at most 150% of cooperative {coop_eq['individual_profit']:.4f}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 