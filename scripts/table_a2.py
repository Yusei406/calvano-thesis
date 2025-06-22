#!/usr/bin/env python3
"""
Simplified Table A.2 replication script for Calvano et al. (2020).
"""

import argparse
import numpy as np
import sys
import os
import time
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import multi_seed_training
from env import DemandEnvironment


def main():
    parser = argparse.ArgumentParser(description='Calvano Table A.2 Replication')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per seed (default: 1000)')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Number of random seeds (default: 5)')
    parser.add_argument('--csv', type=str, default='results/table_a2_results.csv',
                       help='CSV output file')
    parser.add_argument('--json', type=str, default='results/table_a2_results.json',
                       help='JSON output file')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.csv).parent.mkdir(exist_ok=True)
    Path(args.json).parent.mkdir(exist_ok=True)
    
    print("🎯 Calvano et al. (2020) Table A.2 Replication")
    print("=" * 60)
    print(f"Episodes per seed: {args.episodes:,}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"CSV output: {args.csv}")
    print(f"JSON output: {args.json}")
    print()
    
    # Environment info
    env = DemandEnvironment()
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    print(f"Environment Setup:")
    print(f"  Nash equilibrium: π={nash_eq['individual_profit']:.4f}")
    print(f"  Cooperative equilibrium: π={coop_eq['individual_profit']:.4f}")
    print()
    
    # Run multi-seed training
    start_time = time.time()
    
    results = multi_seed_training(
        n_seeds=args.n_seeds,
        n_episodes=args.episodes,
        iterations_per_episode=25000,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon_decay_beta=9.21e-5,
        memory_length=1
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract results
    individual_mean = results['stats']['individual_mean']
    individual_std = results['stats']['individual_std']
    joint_mean = results['stats']['joint_mean']
    joint_std = results['stats']['joint_std']
    
    # Table A.2 targets
    target_individual = 0.18
    target_joint = 0.26
    
    print(f"\n" + "=" * 60)
    print(f"🏁 RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Training completed in {elapsed_time/60:.1f} minutes")
    print()
    
    print(f"📊 Performance vs. Calvano Table A.2:")
    print(f"  Individual profit: {individual_mean:.4f} ± {individual_std:.4f}")
    print(f"    Target: {target_individual:.4f}")
    print(f"    Ratio: {individual_mean/target_individual:.2f}x")
    print()
    
    print(f"  Joint profit: {joint_mean:.4f} ± {joint_std:.4f}")
    print(f"    Target: {target_joint:.4f}")
    print(f"    Ratio: {joint_mean/target_joint:.2f}x")
    print()
    
    # Create output data
    output_data = {
        'individual_mean': individual_mean,
        'individual_std': individual_std,
        'joint_mean': joint_mean,
        'joint_std': joint_std,
        'target_individual': target_individual,
        'target_joint': target_joint,
        'n_seeds': args.n_seeds,
        'episodes': args.episodes,
        'elapsed_time_minutes': elapsed_time/60
    }
    
    # Save JSON
    with open(args.json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame([output_data])
    df.to_csv(args.csv, index=False)
    
    print(f"💾 Results saved to:")
    print(f"   - {args.json}")
    print(f"   - {args.csv}")
    
    print(f"\n✅ Table A.2 replication complete!")


if __name__ == "__main__":
    main()
