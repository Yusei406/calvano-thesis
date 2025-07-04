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
        epsilon_decay_beta: float = 4e-6,  # β = 4×10^-6 from paper
        iterations_per_episode: int = 25000,  # For beta normalization
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
        
        # Beta calculated for ε(25000) ≈ 0.1: β = -ln(0.1)/25000 ≈ 9.21×10^-5
        # ε(t) = exp(-βt) where β per iteration
        self.beta = epsilon_decay_beta
        self.iterations_per_episode = iterations_per_episode
        
        self.memory_length = memory_length
        self.grid_size = grid_size
        self.grid_extension = grid_extension
        
        self.rng = np.random.RandomState(rng_seed)
        
        # Validation: iterations_per_episode should be >= 25000 for Table A.2
        if self.iterations_per_episode < 25000:
            import warnings
            warnings.warn(
                f"iterations_per_episode={self.iterations_per_episode} < 25000. "
                f"Table A.2 replication requires 25,000 iterations per episode.",
                UserWarning
            )
        
        # Initialize dynamic price grid
        self._initialize_price_grid()
        
        # State and action spaces
        self.n_actions = len(self.price_grid)
        self.n_states = self.n_actions ** (self.env.n_agents * self.memory_length)
        
        # Q-table: Initialize with zeros as per Appendix F
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
            'beta': self.beta,
            'iterations_per_episode': self.iterations_per_episode
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
        
    def step(self, opponent_prices: np.ndarray) -> float:
        """Select a price for the current iteration.

        Parameters
        ----------
        opponent_prices : np.ndarray
            Current prices of all agents. The agent encodes the state using
            the opponent's price from the previous period.

        Returns
        -------
        float
            The price chosen by the agent for this iteration.
        """
        # Encode current state
        current_state = self._encode_state(opponent_prices)

        # Select action
        action = self.select_action(current_state)
        selected_price = self._index_to_price(action)

        # Store state-action pair for learning after reward is observed
        self.last_state = current_state
        self.last_action = action

        # Update epsilon after each action
        self.update_epsilon()

        return selected_price
    
    def learn(self, reward: float, next_opponent_prices: np.ndarray):
        """Update the Q-table using the observed reward and next state."""
        if hasattr(self, 'last_state'):
            next_state = self._encode_state(next_opponent_prices)
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
        self.learn(rewards[self.agent_id], observation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'iteration_count': self.iteration_count,
            'episode_count': self.episode_count,
            'current_epsilon': self.current_epsilon,
            'beta': self.beta,
            'iterations_per_episode': self.iterations_per_episode,
            'q_table_mean': float(np.mean(self.q_table)),
            'q_table_std': float(np.std(self.q_table)),
            'equilibrium_info': self.equilibrium_info
        } 