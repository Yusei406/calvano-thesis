#!/usr/bin/env python3
"""
Table A.2 replication script for Calvano et al. (2020) with parallel and resume support.
"""

import argparse
import numpy as np
import sys
import os
import time
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_agents
from env import DemandEnvironment


def run_single_seed(args_tuple):
    """Run training for a single seed (for parallel execution)"""
    seed, episodes, iterations_per_episode, output_dir = args_tuple
    
    try:
        print(f"🏃 Starting seed {seed}")
        
        # Check if this seed is already completed
        seed_file = Path(output_dir) / f"seed_{seed}.csv"
        if seed_file.exists():
            print(f"✅ Seed {seed} already completed, skipping")
            return seed, None
        
        # Run training
        agents, env, history = train_agents(
            n_episodes=episodes,
            iterations_per_episode=iterations_per_episode,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon_decay_beta=9.21e-5,
            memory_length=1,
            rng_seed=seed,
            verbose=False
        )
        
        # Calculate final profits
        final_individual = history['individual_profits'][-1]
        final_joint = history['joint_profits'][-1]
        
        # Save individual seed result
        seed_result = {
            'seed': seed,
            'individual_profit': final_individual,
            'joint_profit': final_joint,
            'episodes': episodes,
            'iterations_per_episode': iterations_per_episode
        }
        
        # Save to individual file
        pd.DataFrame([seed_result]).to_csv(seed_file, index=False)
        print(f"✅ Seed {seed} completed: Individual {seed_result['individual_profit']:.4f}")
        
        return seed, seed_result
        
    except Exception as e:
        print(f"❌ Seed {seed} failed: {e}")
        return seed, None


def main():
    parser = argparse.ArgumentParser(description='Calvano Table A.2 Replication with Parallel Support')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per seed (default: 1000)')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Number of seeds (default: 5)')
    parser.add_argument('--iterations-per-episode', type=int, default=25000,
                       help='Iterations per episode (default: 25000)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing partial results')
    parser.add_argument('--out-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--csv', type=str, default='results/table_a2_parallel.csv',
                       help='CSV output file')
    parser.add_argument('--json', type=str, default='results/table_a2_parallel.json',
                       help='JSON output file')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.out_dir).mkdir(exist_ok=True)
    
    print("🎯 Calvano et al. (2020) Table A.2 Replication (Parallel)")
    print("=" * 60)
    print(f"Episodes per seed: {args.episodes:,}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Parallel workers: {args.workers}")
    print(f"Resume mode: {args.resume}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Environment setup
    env = DemandEnvironment()
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    print("Environment Setup:")
    print(f"  Nash equilibrium: π={nash_eq['individual_profit']:.4f}")
    print(f"  Cooperative equilibrium: π={coop_eq['individual_profit']:.4f}")
    print()
    
    # Determine which seeds to run
    if args.resume:
        # Check existing results
        existing_seeds = []
        for seed in range(args.n_seeds):
            seed_file = Path(args.out_dir) / f"seed_{seed}.csv"
            if seed_file.exists():
                existing_seeds.append(seed)
        
        remaining_seeds = [s for s in range(args.n_seeds) if s not in existing_seeds]
        print(f"📁 Resume mode: {len(existing_seeds)} seeds completed, {len(remaining_seeds)} remaining")
        seeds_to_run = remaining_seeds
    else:
        seeds_to_run = list(range(args.n_seeds))
        print(f"🚀 Fresh start: Running all {len(seeds_to_run)} seeds")
    
    if not seeds_to_run:
        print("✅ All seeds already completed!")
    else:
        # Prepare arguments for parallel execution
        worker_args = [(seed, args.episodes, args.iterations_per_episode, args.out_dir) 
                      for seed in seeds_to_run]
        
        # Run parallel training
        start_time = time.time()
        results = {}
        
        if args.workers == 1:
            # Sequential execution
            for args_tuple in worker_args:
                seed, result = run_single_seed(args_tuple)
                if result:
                    results[seed] = result
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_seed = {executor.submit(run_single_seed, args_tuple): args_tuple[0] 
                                for args_tuple in worker_args}
                
                for future in as_completed(future_to_seed):
                    seed, result = future.result()
                    if result:
                        results[seed] = result
        
        elapsed_time = time.time() - start_time
        print(f"\n⏰ Training completed in {elapsed_time/60:.1f} minutes")
    
    # Collect all results (including existing ones if resume)
    all_results = []
    for seed in range(args.n_seeds):
        seed_file = Path(args.out_dir) / f"seed_{seed}.csv"
        if seed_file.exists():
            seed_data = pd.read_csv(seed_file).iloc[0]
            all_results.append({
                'seed': int(seed_data['seed']),
                'individual_profit': float(seed_data['individual_profit']),
                'joint_profit': float(seed_data['joint_profit'])
            })
    
    if len(all_results) == args.n_seeds:
        # Calculate statistics
        individual_profits = [r['individual_profit'] for r in all_results]
        joint_profits = [r['joint_profit'] for r in all_results]
        
        individual_mean = np.mean(individual_profits)
        individual_std = np.std(individual_profits, ddof=1)
        joint_mean = np.mean(joint_profits)
        joint_std = np.std(joint_profits, ddof=1)
        
        # Display results
        print(f"\n📊 Multi-seed Results (n={len(all_results)}):")
        for result in all_results:
            print(f"Seed {result['seed']}: Individual {result['individual_profit']:.4f}, Joint {result['joint_profit']:.4f}")
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Individual: {individual_mean:.4f} ± {individual_std:.4f}")
        print(f"   Joint: {joint_mean:.4f} ± {joint_std:.4f}")
        print(f"   Nash ratio: {individual_mean/nash_eq['individual_profit']*100:.1f}%")
        
        # Save aggregated results
        summary = {
            'individual_mean': individual_mean,
            'individual_std': individual_std,
            'joint_mean': joint_mean,
            'joint_std': joint_std,
            'target_individual': 0.18,
            'target_joint': 0.26,
            'n_seeds': len(all_results),
            'episodes': args.episodes,
            'elapsed_time_minutes': elapsed_time/60 if 'elapsed_time' in locals() else None
        }
        
        # Save to CSV and JSON
        pd.DataFrame([summary]).to_csv(args.csv, index=False)
        
        with open(args.json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to:")
        print(f"   - {args.json}")
        print(f"   - {args.csv}")
        print(f"   - Individual seed files in {args.out_dir}/")
        
        print("\n✅ Table A.2 replication complete!")
    else:
        print(f"⚠️  Only {len(all_results)}/{args.n_seeds} seeds completed")


if __name__ == "__main__":
    main()
