#!/usr/bin/env python3
"""
Main reproduction script for Calvano et al. (2020).
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import multi_seed_training

def main():
    parser = argparse.ArgumentParser(description='Calvano (2020) Reproduction')
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--fast', action='store_true', help='Fast mode: 1000 episodes')
    
    args = parser.parse_args()
    
    if args.fast:
        args.episodes = 1000
        args.n_seeds = 3
        print("🚀 Fast mode enabled")
    
    print(f"🎯 Calvano Reproduction")
    print(f"Episodes: {args.episodes:,}, Seeds: {args.n_seeds}")
    
    # Fixed seeds for reproducibility
    seeds = [1337, 2021, 2025, 4242, 9999, 1234, 5678, 8888, 7777, 3333][:args.n_seeds]
    
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        results = multi_seed_training(
            n_seeds=args.n_seeds,
            seeds=seeds,
            n_episodes=args.episodes,
            iterations_per_episode=25000
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        stats = results['summary_stats']
        individual_mean = stats['individual_profit']['mean']
        joint_mean = stats['joint_profit']['mean']
        
        # Check targets
        individual_pass = 0.15 <= individual_mean <= 0.21
        joint_pass = 0.22 <= joint_mean <= 0.30
        overall_pass = individual_pass and joint_pass
        
        print(f"\n🏁 RESULTS ({elapsed_time/3600:.2f} hours)")
        print(f"Individual: {individual_mean:.4f} {'✅' if individual_pass else '❌'}")
        print(f"Joint: {joint_mean:.4f} {'✅' if joint_pass else '❌'}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/reproduction_{timestamp}.json"
        
        output = {
            'config': {'episodes': args.episodes, 'seeds': args.n_seeds},
            'results': {
                'indiv_mean': float(individual_mean),
                'joint_mean': float(joint_mean),
                'individual_pass': individual_pass,
                'joint_pass': joint_pass,
                'overall_pass': overall_pass
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved: {filename}")
        print(f"�� Status: {'SUCCESS' if overall_pass else 'PARTIAL'}")
        
        sys.exit(0 if overall_pass else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
