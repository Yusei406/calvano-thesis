"""
Dynamic price grid generation for Calvano et al. (2020).
"""

import numpy as np
from typing import Tuple


def make_grid(p_nash: float, p_coop: float, xi: float = 0.1, m: int = 15) -> np.ndarray:
    """
    Generate dynamic price grid according to Calvano et al. (2020).
    
    Args:
        p_nash: Nash equilibrium price
        p_coop: Cooperative (monopoly) price  
        xi: Grid extension parameter (default: 0.1)
        m: Number of grid points (default: 15)
        
    Returns:
        Price grid as numpy array
        
    Grid specification:
        Range: [p_N - ξ(p_M - p_N), p_M + ξ(p_M - p_N)]
        Points: m equally spaced points
    """
    # Ensure proper ordering
    if p_nash > p_coop:
        p_nash, p_coop = p_coop, p_nash
    
    # Grid boundaries
    grid_width = p_coop - p_nash
    lo = p_nash - xi * grid_width
    hi = p_coop + xi * grid_width
    
    # Generate grid
    price_grid = np.linspace(lo, hi, m)
    
    return price_grid


def validate_grid(price_grid: np.ndarray, p_nash: float, p_coop: float) -> Tuple[bool, dict]:
    """
    Validate that price grid meets Calvano specifications.
    
    Args:
        price_grid: Generated price grid
        p_nash: Nash equilibrium price
        p_coop: Cooperative price
        
    Returns:
        (is_valid, diagnostics)
    """
    diagnostics = {
        'grid_size': len(price_grid),
        'grid_min': price_grid.min(),
        'grid_max': price_grid.max(),
        'nash_in_grid': p_nash >= price_grid.min() and p_nash <= price_grid.max(),
        'coop_in_grid': p_coop >= price_grid.min() and p_coop <= price_grid.max(),
        'grid_spacing': np.diff(price_grid).std() < 1e-10,  # Uniform spacing check
        'grid_range': price_grid.max() - price_grid.min()
    }
    
    # Validation criteria
    is_valid = (
        diagnostics['grid_size'] == 15 and
        diagnostics['nash_in_grid'] and
        diagnostics['coop_in_grid'] and
        diagnostics['grid_spacing']  # Uniform spacing
    )
    
    return is_valid, diagnostics


def get_price_index(price: float, price_grid: np.ndarray) -> int:
    """
    Find closest price grid index for a given price.
    
    Args:
        price: Target price
        price_grid: Price grid array
        
    Returns:
        Closest grid index
    """
    return np.argmin(np.abs(price_grid - price))


def grid_stats(price_grid: np.ndarray) -> dict:
    """
    Compute price grid statistics.
    
    Args:
        price_grid: Price grid array
        
    Returns:
        Dictionary with grid statistics
    """
    return {
        'size': len(price_grid),
        'min': price_grid.min(),
        'max': price_grid.max(),
        'mean': price_grid.mean(),
        'range': price_grid.max() - price_grid.min(),
        'spacing': np.mean(np.diff(price_grid)),
        'spacing_std': np.std(np.diff(price_grid))
    }
