#!/usr/bin/env python3
"""
Generate Figure 2b from Calvano et al. (2020) - Price Convergence Plot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_agents
from env import DemandEnvironment

def main():
    parser = argparse.ArgumentParser(description='Generate Calvano Figure 2b')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--out', type=str, default='figs/figure2b.png')
    parser.add_argument('--fast', action='store_true', help='Fast mode: 500 episodes')
    
    args = parser.parse_args()
    
    if args.fast:
        args.episodes = 500
        print("🚀 Fast mode enabled")
    
    # Create output directory
    Path(args.out).parent.mkdir(exist_ok=True)
    
    print(f"📊 Generating Figure 2b")
    print(f"Episodes: {args.episodes:,}")
    print(f"Output: {args.out}")
    
    # Get reference prices
    env = DemandEnvironment()
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    nash_price = nash_eq['prices'][0]
    coop_price = coop_eq['prices'][0]
    
    print(f"Nash price: {nash_price:.3f}")
    print(f"Cooperative price: {coop_price:.3f}")
    
    # Run training
    print("🏃 Running training...")
    agents, env_trained, history = train_agents(
        n_episodes=args.episodes,
        iterations_per_episode=5000,  # Shorter for figure generation
        rng_seed=1337,
        verbose=False
    )
    
    # Extract data
    episodes = history['episodes']
    profits = history['individual_profits']
    epsilon_values = history['epsilon_values']
    
    # Estimate prices from profits (simple mapping)
    nash_profit = nash_eq['individual_profit']
    coop_profit = coop_eq['individual_profit']
    
    estimated_prices = []
    for profit in profits:
        if profit <= nash_profit:
            # Map to [marginal_cost, nash_price]
            ratio = max(profit / nash_profit, 0) if nash_profit > 0 else 0
            price = env.c + ratio * (nash_price - env.c)
        else:
            # Map to [nash_price, coop_price]
            excess = profit - nash_profit
            max_excess = coop_profit - nash_profit
            if max_excess > 0:
                ratio = min(excess / max_excess, 1.0)
                price = nash_price + ratio * (coop_price - nash_price)
            else:
                price = nash_price
        estimated_prices.append(max(price, env.c))  # Ensure price >= marginal cost
    
    # Create figure
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Top panel: Prices
    ax1.plot(episodes, estimated_prices, 'b-', linewidth=1.5, alpha=0.8, label='Agent Prices')
    ax1.axhline(y=nash_price, color='red', linestyle='--', linewidth=2, label=f'Nash ({nash_price:.2f})')
    ax1.axhline(y=coop_price, color='green', linestyle='--', linewidth=2, label=f'Cooperative ({coop_price:.2f})')
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Figure 2b: Price Convergence in Algorithmic Pricing', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Epsilon
    ax2.plot(episodes, epsilon_values, 'purple', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Exploration Rate (ε)', fontsize=12)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    
    print(f"💾 Figure saved: {args.out}")
    
    # Summary
    if estimated_prices:
        final_price = estimated_prices[-1]
        print(f"Final price: {final_price:.4f}")
        print(f"Price/Nash ratio: {final_price/nash_price:.2f}")
    
    print("✅ Figure 2b generation complete!")

if __name__ == "__main__":
    main()
