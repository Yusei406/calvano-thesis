"""
Demand environment for Calvano et al. (2020) duopoly pricing model.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class DemandEnvironment:
    """
    Duopoly demand environment with logit demand function.
    
    Implements the exact demand model from Calvano et al. (2020):
    - Logit demand: D_i = exp((a_i - p_i) / μ) / (exp(a_0/μ) + sum_j exp((a_j - p_j)/μ))
    - Parameters: a_0 = 0, a_i = 2, μ = 0.25, c = 1
    - Prices in currency units 0-100 
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        demand_intercept: float = 0.0,  # a_0 (outside option)
        product_quality: float = 2.0,   # a_i (product quality)
        demand_slope: float = 0.25,     # μ (price sensitivity) 
        marginal_cost: float = 1.0,     # c (marginal cost)
        outside_option: bool = True,
        rng_seed: Optional[int] = None
    ):
        """Initialize demand environment with exact Calvano parameters."""
        self.n_agents = n_agents
        self.a_0 = demand_intercept  # Outside option utility
        self.a_i = product_quality   # Product quality (same for all products)
        self.mu = demand_slope       # Price sensitivity parameter
        self.c = marginal_cost       # Marginal cost
        self.outside_option = outside_option
        
        self.rng = np.random.RandomState(rng_seed)
        self.current_prices = np.zeros(n_agents)
        self.episode_step = 0
        
    def compute_demands(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute logit demands with exact Calvano specification:
        D_i = exp((a_i - p_i) / μ) / (exp(a_0/μ) + sum_j exp((a_j - p_j)/μ))
        """
        # Convert to numpy array if needed
        prices = np.asarray(prices, dtype=float)
        
        # Product quality parameter (identical for both firms)
        a_i = self.a_i
        
        # Product utilities: (a_i - p_i) / μ
        utilities = (a_i - prices) / self.mu
        exp_utilities = np.exp(utilities)
        
        if self.outside_option:
            # Outside option utility: a_0 / μ
            outside_utility = np.exp(self.a_0 / self.mu)
            denominator = outside_utility + np.sum(exp_utilities)
        else:
            denominator = np.sum(exp_utilities)
        
        demands = exp_utilities / denominator
        return demands
    
    def compute_profits(self, prices: np.ndarray) -> np.ndarray:
        """Compute profits: π_i = (p_i - c) * D_i(p)"""
        # Type safety: ensure prices is numpy array
        prices = np.asarray(prices, dtype=float)
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
    
    def compute_nash_equilibrium(self) -> float:
        """
        Compute Nash equilibrium price numerically with high precision.
        For symmetric duopoly with logit demand.
        """
        from scipy.optimize import root
        
        def nash_foc(p):
            """First-order condition for Nash equilibrium."""
            # Symmetric case: both firms charge price p
            prices = np.array([p, p])
            demands = self.compute_demands(prices)
            
            # FOC: ∂π_i/∂p_i = D_i + (p_i - c) * ∂D_i/∂p_i = 0
            # For logit: ∂D_i/∂p_i = -D_i * (1 - D_i) / μ
            d_i = demands[0]
            derivative = -d_i * (1 - d_i) / self.mu
            foc = d_i + (p - self.c) * derivative
            return foc
        
        # Initial guess
        p_initial = self.c + self.mu
        try:
            result = root(nash_foc, p_initial, method='hybr', tol=1e-8)
            if result.success:
                p_nash = result.x[0]
            else:
                # Fallback approximation
                p_nash = self.c + self.mu / 2.0
        except:
            # Fallback approximation
            p_nash = self.c + self.mu / 2.0
        
        return p_nash
    
    def compute_monopoly_price(self) -> float:
        """
        Compute monopoly price numerically with high precision.
        Joint profit maximization for symmetric case using minimize_scalar.
        """
        from scipy.optimize import minimize_scalar
        
        def negative_joint_profit(p):
            """Negative of joint profit for minimization."""
            prices = np.array([p, p])
            profits = self.compute_profits(prices)
            return -np.sum(profits)  # Negative for minimization
        
        try:
            # Use minimize_scalar for joint profit maximization
            result = minimize_scalar(
                negative_joint_profit,
                bounds=[self.c + 0.01, self.c + 15.0],
                method='bounded',
                options={'xatol': 1e-10}
            )
            
            if result.success:
                p_monopoly = result.x
            else:
                # Fallback: higher than Nash
                p_nash = self.compute_nash_equilibrium()
                p_monopoly = p_nash + 1.0
        except:
            # Fallback: higher than Nash
            p_nash = self.compute_nash_equilibrium()
            p_monopoly = p_nash + 1.0
        
        return p_monopoly
    
    def get_nash_equilibrium(self) -> Dict[str, Any]:
        """Compute Nash equilibrium for comparison."""
        nash_price = self.compute_nash_equilibrium()
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
        coop_price = self.compute_monopoly_price()
        coop_prices = np.full(self.n_agents, coop_price)
        coop_profits = self.compute_profits(coop_prices)
        
        return {
            'prices': coop_prices,
            'profits': coop_profits,
            'individual_profit': coop_profits[0],
            'joint_profit': np.sum(coop_profits)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        nash = self.get_nash_equilibrium()
        coop = self.get_collusive_outcome()
        
        return {
            'n_agents': self.n_agents,
            'demand_params': {'a_0': self.a_0, 'a_i': self.a_i, 'mu': self.mu, 'c': self.c},
            'current_prices': self.current_prices.copy(),
            'current_profits': self.compute_profits(self.current_prices),
            'nash_equilibrium': nash,
            'collusive_outcome': coop,
            'episode_step': self.episode_step
        }
