"""
Q-learning agent for Calvano et al. (2020) algorithmic pricing.
"""

import numpy as np
import math
from typing import Optional, Dict, Any, Tuple
from .grid import make_grid


class QLearningAgent:
    """
    Q-learning agent implementing Calvano et al. (2020) specifications.
    
    Key features:
    - Dynamic price grid based on Nash/Cooperative equilibria
    - Iteration-based exponential epsilon decay: ε(t) = exp(-βt)
    - Memory length k=1 (single period state encoding)
    - 15-point price grid with ξ=0.1 extension
    - Beta normalization: β_effective = β_raw / iterations_per_episode
    """
    
    def __init__(
        self,
        agent_id: int,
        env,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon_initial: float = 1.0,
        epsilon_decay_beta: float = 4e-6,  # β = 4×10^-6 (正確な論文値)
        iterations_per_episode: int = 25000,  # 論文推奨値
        memory_length: int = 1,             # k = 1 period
        grid_size: int = 15,                # m = 15 points
        grid_extension: float = 0.1,        # ξ = 0.1
        rng_seed: Optional[int] = None
    ):
        """Initialize Q-learning agent with exact Calvano et al. (2020) specifications."""
        self.agent_id = agent_id
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon_initial = epsilon_initial
        
        # 正確な論文仕様: β* = β × iterations_per_episode
        # β = 4×10^-6, iterations_per_episode = 25000 → β* = 0.1
        self.beta_raw = epsilon_decay_beta
        self.iterations_per_episode = iterations_per_episode
        self.beta_scaled = epsilon_decay_beta * iterations_per_episode  # β* = 0.1
        
        self.memory_length = memory_length
        self.grid_size = grid_size
        self.grid_extension = grid_extension
        
        self.rng = np.random.RandomState(rng_seed)
        
        # 論文仕様確認: iterations_per_episode = 25000
        if self.iterations_per_episode != 25000:
            import warnings
            warnings.warn(
                f"iterations_per_episode={self.iterations_per_episode} != 25000. "
                f"Paper specification recommends 25,000 iterations per episode.",
                UserWarning
            )
        
        # Initialize dynamic price grid
        self._initialize_price_grid()
        
        # State and action spaces
        self.n_actions = len(self.price_grid)
        self.n_states = self.n_actions ** (self.env.n_agents * self.memory_length)
        
        # Q-table: Initialize with uniform random opponent assumption (paper equation 8)
        self.q_table = self._initialize_q_table()
        
        # Learning tracking
        self.iteration_count = 0
        self.episode_count = 0
        self.current_epsilon = self.epsilon_initial
        
        # State memory
        self.state_history = []
        self.reset_state_memory()
        
    def _initialize_price_grid(self):
        """Initialize dynamic price grid based on equilibrium prices."""
        # Compute equilibrium prices using correct environment methods
        nash_eq = self.env.get_nash_equilibrium()
        coop_eq = self.env.get_collusive_outcome()
        p_nash = nash_eq['prices'][0]  # Extract price from equilibrium result
        p_coop = coop_eq['prices'][0]  # Extract price from collusive result
        
        # Generate dynamic grid
        self.price_grid = make_grid(
            p_nash=p_nash,
            p_coop=p_coop,
            xi=self.grid_extension,
            m=self.grid_size
        )
        
        # Store equilibrium info
        self.equilibrium_info = {
            'nash_price': p_nash,
            'cooperative_price': p_coop,
            'grid_range': [self.price_grid.min(), self.price_grid.max()],
            'grid_spacing': np.mean(np.diff(self.price_grid)),
            'beta': self.beta_scaled,
            'iterations_per_episode': self.iterations_per_episode
        }
        
    def _initialize_q_table(self):
        """Initialize Q-table with uniform random opponent assumption."""
        return np.random.uniform(0, 1, (self.n_states, self.n_actions))
        
    def reset_state_memory(self):
        """Reset state memory for new episode."""
        # Initialize with neutral prices (Nash equilibrium)
        initial_price_idx = self._price_to_index(self.equilibrium_info['nash_price'])
        
        # Memory for k=1 period: [opponent_price_t-1]
        self.state_history = [initial_price_idx] * (self.env.n_agents * self.memory_length)
        
    def _price_to_index(self, price: float) -> int:
        """Convert price to grid index."""
        return np.argmin(np.abs(self.price_grid - price))
        
    def _index_to_price(self, index: int) -> float:
        """Convert grid index to price."""
        return self.price_grid[index]
        
    def _encode_state(self, opponent_prices: np.ndarray) -> int:
        """
        Encode state using k=1 period of opponent price history.
        
        State encoding for k=1:
        - state = opponent_price_index (previous period)
        - For n_agents=2: state ∈ {0, 1, ..., 14} (15 possible states)
        """
        # For duopoly with k=1: state is opponent's price index from previous period
        if self.env.n_agents == 2:
            # Get opponent's index
            opponent_idx = 1 - self.agent_id
            # Convert opponent's current price to grid index
            opponent_price_idx = self._price_to_index(opponent_prices[opponent_idx])
            return opponent_price_idx
        else:
            # General case: encode all opponent price indices
            state = 0
            for i, price in enumerate(opponent_prices):
                if i != self.agent_id:  # Skip own price
                    price_idx = self._price_to_index(price)
                    state = state * self.n_actions + price_idx
            return state
    
    def update_epsilon(self):
        """
        Update epsilon using iteration-based exponential decay: ε(t) = exp(-βt)
        
        改善版: より適切なβ値の計算
        - 目標: 最終エピソードでε ≈ 0.01-0.05 (完全に0にならないように)
        - β調整: エピソード終了時点でのε値を考慮
        """
        self.iteration_count += 1
        
        # Calvano et al. (2020) spec:
        # ε(t) = exp(-β * t) where β is the *raw* coefficient (4e-6).
        # β* = β × iterations_per_episode is only a descriptive statistic, not
        # used directly in the decay formula. Using beta_scaled here causes an
        # extra multiplication by iterations_per_episode, making ε collapse to
        # the minimum after a few hundred iterations. We therefore use
        # beta_raw for the decay and drop the ad-hoc scaling.

        self.current_epsilon = math.exp(-self.beta_raw * self.iteration_count)
        
        # 最小値を設定してε=0になることを防ぐ
        min_epsilon = 0.001  # 最小値1%
        self.current_epsilon = max(self.current_epsilon, min_epsilon)
        
    def select_action(self, state: int) -> int:
        """Select action using ε-greedy policy with random tie-breaking."""
        if self.rng.random() < self.current_epsilon:
            # Exploration: random action
            action = self.rng.randint(0, self.n_actions)
        else:
            # Exploitation: best action with random tie-breaking
            q_values = self.q_table[state, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = self.rng.choice(best_actions)
            
        return action
    
    def update_q_table(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int
    ):
        """Update Q-table using Q-learning rule."""
        best_next_q = np.max(self.q_table[next_state, :])
        current_q = self.q_table[state, action]
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state, action] = new_q
        
    def step(self, opponent_prices: np.ndarray, rewards: np.ndarray) -> float:
        """
        Execute one learning step.
        
        Args:
            opponent_prices: Current prices of all agents
            rewards: Rewards for all agents
            
        Returns:
            Selected price for this agent
        """
        # Encode current state
        current_state = self._encode_state(opponent_prices)
        
        # Select action
        action = self.select_action(current_state)
        selected_price = self._index_to_price(action)
        
        # Store experience for potential Q-table update
        self.last_state = current_state
        self.last_action = action
        self.last_reward = rewards[self.agent_id]
        
        # Update epsilon (iteration_count incremented inside update_epsilon)
        self.update_epsilon()
        
        return selected_price
    
    def learn(self, rewards: np.ndarray, next_opponent_prices: np.ndarray):
        """Update Q-table using reward from the last action."""
        if hasattr(self, 'last_state'):
            next_state = self._encode_state(next_opponent_prices)
            reward = rewards[self.agent_id]
            self.update_q_table(
                self.last_state,
                self.last_action,
                reward,
                next_state
            )
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.reset_state_memory()
    
    def get_action(self, observation, agent_id, explore=True):
        """Get action for compatibility with existing interface."""
        if not explore:
            # Exploitation only
            state = self._encode_state(observation)
            action = np.argmax(self.q_table[state, :])
        else:
            # Normal ε-greedy
            state = self._encode_state(observation)
            action = self.select_action(state)
        
        return action
    
    def update(self, observation, actions, rewards):
        """Update agent for compatibility with existing interface."""
        # Update Q-table
        self.learn(rewards, observation)

        # Update epsilon
        self.update_epsilon()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'iteration_count': self.iteration_count,
            'episode_count': self.episode_count,
            'current_epsilon': self.current_epsilon,
            'beta': self.beta_scaled,
            'iterations_per_episode': self.iterations_per_episode,
            'q_table_mean': float(np.mean(self.q_table)),
            'q_table_std': float(np.std(self.q_table)),
            'equilibrium_info': self.equilibrium_info
        } 