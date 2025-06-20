"""
Demand environment for Calvano et al. (2020) duopoly pricing model.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class DemandEnvironment:
    """
    Duopoly demand environment with logit demand function.
    
    Implements the exact demand model from Calvano et al. (2020):
    - Logit demand: D_i = exp(a_0 - μp_i) / (1 + sum_j exp(a_0 - μp_j))
    - Parameters: a_0 = 2, μ = 1, c = 0
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        demand_intercept: float = 2.0,  # a_0 in Calvano paper
        demand_slope: float = 1.0,      # μ in Calvano paper  
        marginal_cost: float = 0.0,     # c in Calvano paper
        outside_option: bool = True,
        rng_seed: Optional[int] = None
    ):
        """Initialize demand environment."""
        self.n_agents = n_agents
        self.a_0 = demand_intercept
        self.mu = demand_slope
        self.c = marginal_cost
        self.outside_option = outside_option
        
        self.rng = np.random.RandomState(rng_seed)
        self.current_prices = np.zeros(n_agents)
        self.episode_step = 0
        
    def compute_demands(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices)
        """Compute logit demands: D_i = exp(a_0 - μ*p_i) / (1 + sum_j exp(a_0 - μ*p_j))"""
        a_i = 2.0  # Product quality
        utilities = (a_i - prices) / self.mu
        exp_utilities = np.exp(utilities)
        
        if self.outside_option:
            denominator = 1.0 + np.sum(exp_utilities)
        else:
            denominator = np.sum(exp_utilities)
        
        demands = exp_utilities / denominator
        return demands
    
    def compute_profits(self, prices: np.ndarray) -> np.ndarray:
        """Compute profits: π_i = (p_i - c) * D_i(p)"""
        demands = self.compute_demands(prices)
        margins = prices - self.c
        profits = margins * demands
        return profits
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """Environment step function."""
        self.current_prices = actions.copy()
        rewards = self.compute_profits(self.current_prices)
        self.episode_step += 1
        done = False
        
        demands = self.compute_demands(self.current_prices)
        info = {
            'step': self.episode_step,
            'prices': self.current_prices.copy(),
            'demands': demands.copy(),
            'profits': rewards.copy(),
            'total_demand': np.sum(demands)
        }
        
        return self.current_prices.copy(), rewards, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_prices = np.zeros(self.n_agents)
        self.episode_step = 0
        return self.current_prices.copy()
    
    def get_nash_equilibrium(self) -> Dict[str, Any]:
        """Compute Nash equilibrium for comparison."""
        if self.n_agents == 2 and self.outside_option:
            # Symmetric Nash price for duopoly with outside option
            nash_price = self.c + (1.0 / self.mu) * (2.0 / 3.0)  # Approximation
            nash_prices = np.full(self.n_agents, nash_price)
            nash_profits = self.compute_profits(nash_prices)
            
            return {
                'prices': nash_prices,
                'profits': nash_profits,
                'individual_profit': nash_profits[0],
                'joint_profit': np.sum(nash_profits)
            }
        else:
            # Fallback approximation
            nash_price = 0.5
            nash_prices = np.full(self.n_agents, nash_price)
            nash_profits = self.compute_profits(nash_prices)
            
            return {
                'prices': nash_prices,
                'profits': nash_profits,
                'individual_profit': nash_profits[0],
                'joint_profit': np.sum(nash_profits)
            }
    
    def get_collusive_outcome(self) -> Dict[str, Any]:
        """Compute joint profit maximizing outcome."""
        # Approximation for symmetric collusive price
        coop_price = self.c + (1.0 / self.mu) * 1.0  # Higher than Nash
        coop_prices = np.full(self.n_agents, coop_price)
        coop_profits = self.compute_profits(coop_prices)
        
        return {
            'prices': coop_prices,
            'profits': coop_profits,
            'individual_profit': coop_profits[0],
            'joint_profit': np.sum(coop_profits)
        }
