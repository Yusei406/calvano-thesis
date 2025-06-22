"""
Q-learning agent for Calvano et al. (2020) algorithmic pricing.
"""

import numpy as np
import math
from typing import Optional, Dict, Any, Tuple
from grid import make_grid


class QLearningAgent:
    """
    Q-learning agent implementing Calvano et al. (2020) specifications.
    
    Key features:
    - Dynamic price grid based on Nash/Cooperative equilibria
    - Iteration-based exponential epsilon decay: ε(t) = exp(-βt)
    - Memory length k=1 (single period state encoding)
    - 15-point price grid with ξ=0.1 extension
    """
    
    def __init__(
        self,
        agent_id: int,
        env,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon_initial: float = 1.0,
        epsilon_decay_beta: float = 9.21e-5,  # β = 4×10^-6 from paper
        memory_length: int = 1,             # k = 1 period
        grid_size: int = 15,                # m = 15 points
        grid_extension: float = 0.1,        # ξ = 0.1
        rng_seed: Optional[int] = None
    ):
        """Initialize Q-learning agent with Calvano specifications."""
        self.agent_id = agent_id
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon_initial = epsilon_initial
        self.beta = epsilon_decay_beta  # Exponential decay parameter
        self.memory_length = memory_length
        self.grid_size = grid_size
        self.grid_extension = grid_extension
        
        self.rng = np.random.RandomState(rng_seed)
        
        # Initialize dynamic price grid
        self._initialize_price_grid()
        
        # State and action spaces
        self.n_actions = len(self.price_grid)
        self.n_states = self.n_actions ** (self.env.n_agents * self.memory_length)
        
        # Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Learning tracking
        self.iteration_count = 0
        self.episode_count = 0
        self.current_epsilon = self.epsilon_initial
        
        # State memory
        self.state_history = []
        self.reset_state_memory()
        
    def _initialize_price_grid(self):
        """Initialize dynamic price grid based on equilibrium prices."""
        # Compute equilibrium prices
        nash_eq = self.env.get_nash_equilibrium()
        p_nash = nash_eq["prices"][0]
        coop_eq = self.env.get_collusive_outcome()
        p_coop = coop_eq["prices"][0]
        
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
            'grid_spacing': np.mean(np.diff(self.price_grid))
        }
        
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
        """Update epsilon using iteration-based exponential decay: ε(t) = exp(-βt)"""
        self.iteration_count += 1
        self.current_epsilon = math.exp(-self.beta * self.iteration_count)
        
    def select_action(self, state: int) -> int:
        """Select action using ε-greedy policy."""
        if self.rng.random() < self.current_epsilon:
            # Exploration: random action
            action = self.rng.randint(0, self.n_actions)
        else:
            # Exploitation: best action
            action = np.argmax(self.q_table[state, :])
            
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
        
        # Update iteration count and epsilon
        self.iteration_count += 1
        self.update_epsilon()
        
        return selected_price
    
    def learn(self, next_opponent_prices: np.ndarray):
        """Update Q-table based on observed transition."""
        if hasattr(self, 'last_state'):
            next_state = self._encode_state(next_opponent_prices)
            self.update_q_table(
                self.last_state,
                self.last_action, 
                self.last_reward,
                next_state
            )
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.reset_state_memory()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        q_stats = {
            'q_mean': np.mean(self.q_table),
            'q_std': np.std(self.q_table),
            'q_max': np.max(self.q_table),
            'q_min': np.min(self.q_table),
            'q_nonzero': np.count_nonzero(self.q_table)
        }
        
        return {
            'agent_id': self.agent_id,
            'iteration_count': self.iteration_count,
            'episode_count': self.episode_count,
            'current_epsilon': self.current_epsilon,
            'learning_rate': self.alpha,
            'discount_factor': self.gamma,
            'price_grid': self.price_grid.tolist(),
            'equilibrium_info': self.equilibrium_info,
            'q_table_stats': q_stats,
            'state_space_size': self.n_states,
            'action_space_size': self.n_actions
        } 