"""
Training script for Calvano et al. (2020) Q-learning implementation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from env import DemandEnvironment
from agent import QLearningAgent


def train_agents(
    n_episodes: int = 2000,
    iterations_per_episode: int = 25000,  # Calvano specification
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    epsilon_decay_beta: float = 4e-6,     # β = 4×10^-6
    memory_length: int = 1,               # k = 1 period
    convergence_window: int = 100,
    convergence_threshold: float = 0.001,
    rng_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[QLearningAgent], DemandEnvironment, Dict[str, Any]]:
    """
    Train Q-learning agents with Calvano et al. (2020) specifications.
    """
    # Initialize environment
    env = DemandEnvironment(
        n_agents=2,
        demand_intercept=0.0,
        demand_slope=0.25,
        marginal_cost=1.0,
        rng_seed=rng_seed
    )
    
    # Initialize agents
    agents = []
    for i in range(env.n_agents):
        agent = QLearningAgent(
            agent_id=i,
            env=env,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_decay_beta=epsilon_decay_beta,
            memory_length=memory_length,
            rng_seed=rng_seed + i if rng_seed else None
        )
        agents.append(agent)
    
    # Training history tracking
    history = {
        'episodes': [],
        'individual_profits': [],
        'joint_profits': [],
        'epsilon_values': [],
        'total_iterations': 0
    }
    
    # Get equilibrium benchmarks
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    if verbose:
        print(f"🎯 Training Configuration:")
        print(f"   Episodes: {n_episodes}")
        print(f"   Iterations per episode: {iterations_per_episode:,}")
        print(f"   Nash individual: {nash_eq['individual_profit']:.4f}")
        print(f"   Cooperative individual: {coop_eq['individual_profit']:.4f}")
        print()
    
    # Training loop
    for episode in range(n_episodes):
        # Reset for new episode
        env.reset()
        for agent in agents:
            agent.reset_episode()
        
        episode_profits = []
        
        # Inner loop: iterations within episode
        for iteration in range(iterations_per_episode):
            current_prices = env.current_prices.copy()
            
            # Agent actions
            new_prices = np.zeros(env.n_agents)
            for i, agent in enumerate(agents):
                new_prices[i] = agent.step(current_prices, env.compute_profits(current_prices))
            
            # Environment step
            _, rewards, _, info = env.step(new_prices)
            
            # Agent learning
            for agent in agents:
                agent.learn(new_prices)
            
            # Track data
            if iteration % 100 == 0:
                episode_profits.append(rewards.copy())
        
        # Episode summary
        episode_avg_profits = np.mean(episode_profits, axis=0)
        episode_individual = episode_avg_profits[0]
        episode_joint = np.sum(episode_avg_profits)
        
        # Update history
        history['episodes'].append(episode)
        history['individual_profits'].append(episode_individual)
        history['joint_profits'].append(episode_joint)
        history['epsilon_values'].append(agents[0].current_epsilon)
        history['total_iterations'] = (episode + 1) * iterations_per_episode
        
        # Progress reporting
        if verbose and (episode % 100 == 0 or episode < 10):
            nash_ratio = episode_individual / nash_eq['individual_profit']
            print(f"Episode {episode:>4d} | Individual: {episode_individual:.4f} ({nash_ratio:.1%} Nash) | ε: {agents[0].current_epsilon:.4f}")
    
    # Final summary
    final_individual = history['individual_profits'][-1]
    final_joint = history['joint_profits'][-1]
    
    training_summary = {
        'episodes_completed': n_episodes,
        'total_iterations': history['total_iterations'],
        'final_individual_profit': final_individual,
        'final_joint_profit': final_joint,
        'nash_ratio_individual': final_individual / nash_eq['individual_profit'],
        'nash_ratio_joint': final_joint / nash_eq['joint_profit'],
        'final_epsilon': agents[0].current_epsilon,
        'nash_benchmark': nash_eq,
        'cooperative_benchmark': coop_eq
    }
    
    history['training_summary'] = training_summary
    
    if verbose:
        print(f"\n🏁 Training Complete:")
        print(f"   Final individual: {final_individual:.4f}")
        print(f"   Nash ratio: {training_summary['nash_ratio_individual']:.1%}")
    
    return agents, env, history


def multi_seed_training(
    n_seeds: int = 5,
    seeds: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run training with multiple random seeds."""
    if seeds is None:
        np.random.seed(42)
        seeds = np.random.randint(0, 10000, n_seeds).tolist()
    
    results = []
    for i, seed in enumerate(seeds):
        print(f"Training seed {i+1}/{n_seeds} (seed={seed})")
        agents, env, history = train_agents(rng_seed=seed, verbose=False, **kwargs)
        results.append({
            'seed': seed,
            'agents': agents,
            'env': env,
            'history': history,
            'summary': history['training_summary']
        })
    
    # Aggregate statistics
    individual_profits = [r['summary']['final_individual_profit'] for r in results]
    joint_profits = [r['summary']['final_joint_profit'] for r in results]
    nash_ratios = [r['summary']['nash_ratio_individual'] for r in results]
    
    return {
        'n_seeds': n_seeds,
        'seeds': seeds,
        'individual_results': results,
        'summary_stats': {
            'individual_profit': {
                'mean': np.mean(individual_profits),
                'std': np.std(individual_profits)
            },
            'joint_profit': {
                'mean': np.mean(joint_profits),
                'std': np.std(joint_profits)
            },
            'nash_ratio_individual': {
                'mean': np.mean(nash_ratios),
                'std': np.std(nash_ratios)
            }
        }
    }
