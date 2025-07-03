"""
Training module for Calvano et al. (2020) Q-learning agents.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional

from .env import DemandEnvironment
from .agent import QLearningAgent


ITERATIONS_PER_EPISODE = 25000  # Calvano specification


def train_agents(
    n_episodes: int = 50000,
    iterations_per_episode: int = ITERATIONS_PER_EPISODE,
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    epsilon_decay_beta: float = 4e-6,
    memory_length: int = 1,
    rng_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[QLearningAgent], DemandEnvironment, Dict[str, Any]]:
    """
    Train Q-learning agents with Calvano et al. (2020) specification.
    """
    # Input validation
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if iterations_per_episode <= 0:
        raise ValueError(f"iterations_per_episode must be positive, got {iterations_per_episode}")
    if not (0 < learning_rate <= 1):
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    if not (0 < discount_factor < 1):
        raise ValueError(f"discount_factor must be in (0, 1), got {discount_factor}")
    if epsilon_decay_beta <= 0:
        raise ValueError(f"epsilon_decay_beta must be positive, got {epsilon_decay_beta}")
    if memory_length <= 0:
        raise ValueError(f"memory_length must be positive, got {memory_length}")
    
    # Environment setup with complete Calvano parameters
    env = DemandEnvironment(
        demand_intercept=0.0,      # a_0 = 0
        product_quality=2.0,       # a_i = 2
        demand_slope=0.25,         # μ = 0.25
        marginal_cost=1.0,         # c = 1
        rng_seed=rng_seed
    )
    
    # Agent setup
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
    
    # Training history
    history = {
        'episodes': [],
        'individual_profits': [],
        'joint_profits': [],
        'epsilon_values': [],
        'total_iterations': 0,
        'beta_info': {
            'beta_raw': epsilon_decay_beta,
            'iterations_per_episode': iterations_per_episode,
            'beta_effective': epsilon_decay_beta / iterations_per_episode,
            'epsilon_at_convergence': np.exp(-epsilon_decay_beta * iterations_per_episode)
        }
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

        # Initial rewards from starting prices
        prev_rewards = env.compute_profits(env.current_prices)
        episode_profits = []

        # Inner loop: iterations within episode
        for iteration in range(iterations_per_episode):
            current_prices = env.current_prices.copy()

            # Agent actions based on previous rewards
            new_prices = np.zeros(env.n_agents)
            for i, agent in enumerate(agents):
                new_prices[i] = agent.step(current_prices, prev_rewards)

            # Environment step to obtain new rewards
            _, rewards, _, info = env.step(new_prices)

            # Agent learning updates with rewards from this step
            for agent in agents:
                agent.learn(rewards, new_prices)

            # Update previous rewards
            prev_rewards = rewards
            
            # Sample episode data
            if iteration % 250 == 0:
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
        'individual_profit': final_individual,
        'joint_profit': final_joint,
        'nash_ratio_individual': final_individual / nash_eq['individual_profit'],
        'nash_ratio_joint': final_joint / nash_eq['joint_profit'],
        'final_epsilon': agents[0].current_epsilon,
        'beta_info': history['beta_info']
    }
    
    if verbose:
        print(f"\n🎯 Training Complete:")
        print(f"   Final individual profit: {final_individual:.4f} ({training_summary['nash_ratio_individual']:.1%} Nash)")
        print(f"   Final joint profit: {final_joint:.4f} ({training_summary['nash_ratio_joint']:.1%} Nash)")
        print(f"   Final epsilon: {agents[0].current_epsilon:.6f}")
    
    history['training_summary'] = training_summary
    
    return agents, env, history


def multi_seed_training(
    n_seeds: int = 5,
    seeds: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run training with multiple random seeds for statistical validation.
    """
    if seeds is None:
        seeds = list(range(n_seeds))
    
    results = {
        'individual_profits': [],
        'joint_profits': [],
        'nash_ratios_individual': [],
        'nash_ratios_joint': [],
        'seeds': seeds
    }
    
    for seed in seeds:
        agents, env, history = train_agents(rng_seed=seed, verbose=False, **kwargs)
        
        final_individual = history['individual_profits'][-1]
        final_joint = history['joint_profits'][-1]
        
        # Get equilibrium for ratios
        nash_eq = env.get_nash_equilibrium()
        
        results['individual_profits'].append(final_individual)
        results['joint_profits'].append(final_joint)
        results['nash_ratios_individual'].append(final_individual / nash_eq['individual_profit'])
        results['nash_ratios_joint'].append(final_joint / nash_eq['joint_profit'])
        
        print(f"Seed {seed}: Individual {final_individual:.4f}, Joint {final_joint:.4f}")
    
    # Summary statistics
    results['stats'] = {
        'individual_mean': np.mean(results['individual_profits']),
        'individual_std': np.std(results['individual_profits']),
        'joint_mean': np.mean(results['joint_profits']),
        'joint_std': np.std(results['joint_profits']),
        'nash_ratio_individual_mean': np.mean(results['nash_ratios_individual']),
        'nash_ratio_joint_mean': np.mean(results['nash_ratios_joint'])
    }
    
    print(f"\n📊 Multi-seed Results (n={n_seeds}):")
    print(f"   Individual: {results['stats']['individual_mean']:.4f} ± {results['stats']['individual_std']:.4f}")
    print(f"   Joint: {results['stats']['joint_mean']:.4f} ± {results['stats']['joint_std']:.4f}")
    print(f"   Nash ratio: {results['stats']['nash_ratio_individual_mean']:.1%}")
    
    return results


if __name__ == "__main__":
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Calvano et al. (2020) Q-learning agents"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=50000,
        help="Number of episodes (default: 50000)"
    )
    
    parser.add_argument(
        "--iterations-per-episode", "-i",
        type=int,
        default=25000,
        help="Iterations per episode (default: 25000)"
    )
    
    parser.add_argument(
        "--learning-rate", "-a",
        type=float,
        default=0.15,
        help="Learning rate (default: 0.15)"
    )
    
    parser.add_argument(
        "--discount-factor", "-g",
        type=float,
        default=0.95,
        help="Discount factor (default: 0.95)"
    )
    
    parser.add_argument(
        "--epsilon-decay-beta", "-b",
        type=float,
        default=4e-6,
        help="Epsilon decay beta (default: 4e-6)"
    )
    
    parser.add_argument(
        "--memory-length", "-m",
        type=int,
        default=1,
        help="Memory length (default: 1)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed (default: None)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Run training
    agents, env, history = train_agents(
        n_episodes=args.episodes,
        iterations_per_episode=args.iterations_per_episode,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_decay_beta=args.epsilon_decay_beta,
        memory_length=args.memory_length,
        rng_seed=args.seed,
        verbose=args.verbose
    )
    
    # Print results
    final_individual = history['training_summary']['individual_profit']
    final_joint = history['training_summary']['joint_profit']
    nash_ratio = history['training_summary']['nash_ratio_individual']
    
    print(f"\n🎯 Training Results:")
    print(f"   Individual profit: {final_individual:.4f}")
    print(f"   Joint profit: {final_joint:.4f}")
    print(f"   Nash ratio: {nash_ratio:.1%}")
    print(f"   Total iterations: {history['total_iterations']:,}") 