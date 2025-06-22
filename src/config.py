"""Configuration loader for Calvano et al. (2020) implementation."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_qlearning_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract Q-learning parameters from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default config.
        
    Returns:
        Q-learning parameters dictionary
    """
    if config is None:
        config = load_config()
    
    qlearning_config = config.get('qlearning', {})
    training_config = config.get('training', {})
    
    return {
        'learning_rate': qlearning_config.get('learning_rate', 0.15),
        'discount_factor': qlearning_config.get('discount_factor', 0.95),
        'epsilon_initial': qlearning_config.get('epsilon_initial', 1.0),
        'epsilon_decay_beta': qlearning_config.get('epsilon_decay_beta', 9.21e-5),
        'memory_length': qlearning_config.get('memory_length', 1),
        'iterations_per_episode': training_config.get('iterations_per_episode', 25000)
    }


def get_environment_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract environment parameters from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default config.
        
    Returns:
        Environment parameters dictionary
    """
    if config is None:
        config = load_config()
    
    env_config = config.get('environment', {})
    
    return {
        'n_agents': env_config.get('n_agents', 2),
        'demand_intercept': env_config.get('demand_intercept', 0.0),
        'demand_slope': env_config.get('demand_slope', 0.25),
        'marginal_cost': env_config.get('marginal_cost', 1.0),
        'product_quality': env_config.get('product_quality', 2.0),
        'outside_option': env_config.get('outside_option', True)
    }


def get_grid_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract grid parameters from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default config.
        
    Returns:
        Grid parameters dictionary
    """
    if config is None:
        config = load_config()
    
    grid_config = config.get('grid', {})
    
    return {
        'size': grid_config.get('size', 15),
        'extension': grid_config.get('extension', 0.1)
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters against Calvano specifications.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Critical parameters that must match paper specifications
    qlearning = config.get('qlearning', {})
    environment = config.get('environment', {})
    training = config.get('training', {})
    
    # Validate Q-learning parameters
    if qlearning.get('discount_factor') != 0.95:
        raise ValueError("discount_factor must be 0.95 per Calvano specification")
    
    if qlearning.get('memory_length') != 1:
        raise ValueError("memory_length must be 1 (k=1) per Calvano specification")
    
    # Validate environment parameters
    if environment.get('demand_intercept') != 0.0:
        raise ValueError("demand_intercept (a_0) must be 0.0 per Calvano specification")
    
    if environment.get('demand_slope') != 0.25:
        raise ValueError("demand_slope (μ) must be 0.25 per Calvano specification")
    
    if environment.get('marginal_cost') != 1.0:
        raise ValueError("marginal_cost (c) must be 1.0 per Calvano specification")
    
    if environment.get('product_quality') != 2.0:
        raise ValueError("product_quality (a_i) must be 2.0 per Calvano specification")
    
    # Validate training parameters for Table A.2 replication
    if training.get('iterations_per_episode', 0) < 25000:
        import warnings
        warnings.warn(
            f"iterations_per_episode={training.get('iterations_per_episode')} < 25000. "
            f"Table A.2 replication requires 25,000 iterations per episode.",
            UserWarning
        )
    
    return True
