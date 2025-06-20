#!/usr/bin/env python3
"""
Table A.2 replication script for Calvano et al. (2020).

Runs multi-seed training to replicate Table A.2 results:
- 50,000+ episodes per seed
- 10+ seeds for statistical validation
- Target: Individual profit ≈ 0.18±0.03, Joint profit ≈ 0.26±0.04
"""

import argparse
import numpy as np
import sys
import os
import time
import json
import csv
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import multi_seed_training
from env import DemandEnvironment


def main():
    parser = argparse.ArgumentParser(description='Calvano Table A.2 Replication')
    parser.add_argument('--episodes', type=int, default=50000,
                       help='Episodes per seed (default: 50000)')
    parser.add_argument('--n-seeds', type=int, default=10,
                       help='Number of random seeds (default: 10)')
    parser.add_argument('--iterations-per-episode', type=int, default=25000,
                       help='Iterations per episode (default: 25000)')
    parser.add_argument('--beta', type=float, default=4e-6,
                       help='Epsilon decay parameter (default: 4e-6)')
    parser.add_argument('--learning-rate', type=float, default=0.15,
                       help='Q-learning rate (default: 0.15)')
    parser.add_argument('--discount-factor', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--csv', type=str, default='results/table_a2_results.csv',
                       help='CSV output file (default: results/table_a2_results.csv)')
    parser.add_argument('--json', type=str, default='results/table_a2_results.json',
                       help='JSON output file (default: results/table_a2_results.json)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Specific seeds to use (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.csv).parent.mkdir(exist_ok=True)
    Path(args.json).parent.mkdir(exist_ok=True)
    
    print("🎯 Calvano et al. (2020) Table A.2 Replication")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Episodes per seed: {args.episodes:,}")
    print(f"  Number of seeds: {args.n_seeds}")
    print(f"  Iterations per episode: {args.iterations_per_episode:,}")
    print(f"  Total iterations per seed: {args.episodes * args.iterations_per_episode:,}")
    print(f"  Epsilon decay β: {args.beta}")
    print(f"  Learning rate α: {args.learning_rate}")
    print(f"  Discount factor γ: {args.discount_factor}")
    print(f"  CSV output: {args.csv}")
    print(f"  JSON output: {args.json}")
    print()
    
    # Environment info
    env = DemandEnvironment(
        demand_intercept=0.0,
        demand_slope=0.25, 
        marginal_cost=1.0
    )
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    print(f"Environment Setup:")
    print(f"  Parameters: a₀=0.0, μ=0.25, c=1.0")
    print(f"  Nash equilibrium: π={nash_eq['individual_profit']:.4f}")
    print(f"  Cooperative equilibrium: π={coop_eq['individual_profit']:.4f}")
    print()
    
    # Run multi-seed training
    start_time = time.time()
    
    try:
        results = multi_seed_training(
            n_seeds=args.n_seeds,
            seeds=args.seeds,
            n_episodes=args.episodes,
            iterations_per_episode=args.iterations_per_episode,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon_decay_beta=args.beta,
            memory_length=1
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        stats = results['summary_stats']
        individual_mean = stats['individual_profit']['mean']
        individual_std = stats['individual_profit']['std']
        joint_mean = stats['joint_profit']['mean']
        joint_std = stats['joint_profit']['std']
        nash_ratio_mean = stats['nash_ratio_individual']['mean']
        nash_ratio_std = stats['nash_ratio_individual']['std']
        convergence_rate = stats['convergence_rate']
        
        # Table A.2 targets
        target_individual = 0.18
        target_individual_tol = 0.03
        target_joint = 0.26
        target_joint_tol = 0.04
        target_convergence = 1.0
        
        print(f"\n" + "=" * 60)
        print(f"🏁 RESULTS SUMMARY")
        print(f"=" * 60)
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print()
        
        print(f"📊 Performance vs. Calvano Table A.2:")
        print(f"  Individual profit: {individual_mean:.4f} ± {individual_std:.4f}")
        print(f"    Target: {target_individual:.4f} ± {target_individual_tol:.4f}")
        
        individual_pass = abs(individual_mean - target_individual) <= target_individual_tol
        print(f"    Status: {'✅ PASS' if individual_pass else '❌ FAIL'}")
        print()
        
        print(f"  Joint profit: {joint_mean:.4f} ± {joint_std:.4f}")
        print(f"    Target: {target_joint:.4f} ± {target_joint_tol:.4f}")
        
        joint_pass = abs(joint_mean - target_joint) <= target_joint_tol
        print(f"    Status: {'✅ PASS' if joint_pass else '❌ FAIL'}")
        print()
        
        print(f"  Convergence rate: {convergence_rate:.1%}")
        print(f"    Target: {target_convergence:.1%}")
        
        convergence_pass = convergence_rate >= 0.9
        print(f"    Status: {'✅ PASS' if convergence_pass else '❌ FAIL'}")
        print()
        
        # Overall assessment
        overall_pass = individual_pass and joint_pass and convergence_pass
        
        print(f"🎯 OVERALL ASSESSMENT:")
        if overall_pass:
            print(f"   🏆 REPLICATION SUCCESSFUL!")
            print(f"   All metrics within Calvano et al. (2020) targets.")
        else:
            print(f"   ⚠️  REPLICATION PARTIAL")
            print(f"   Some metrics outside target ranges.")
        
        # Prepare output data
        output_data = {
            'experiment_config': {
                'episodes': args.episodes,
                'n_seeds': args.n_seeds,
                'iterations_per_episode': args.iterations_per_episode,
                'learning_rate': args.learning_rate,
                'discount_factor': args.discount_factor,
                'epsilon_decay_beta': args.beta,
                'seeds': results['seeds']
            },
            'results': {
                'indiv_mean': float(individual_mean),
                'indiv_std': float(individual_std),
                'joint_mean': float(joint_mean), 
                'joint_std': float(joint_std),
                'nash_ratio_mean': float(nash_ratio_mean),
                'nash_ratio_std': float(nash_ratio_std),
                'convergence_rate': float(convergence_rate),
                'target_individual': target_individual,
                'target_joint': target_joint,
                'individual_pass': individual_pass,
                'joint_pass': joint_pass,
                'convergence_pass': convergence_pass,
                'overall_pass': overall_pass,
                'elapsed_time_hours': elapsed_time / 3600
            },
            'environment': {
                'nash_equilibrium': {
                    'individual_profit': float(nash_eq['individual_profit']),
                    'joint_profit': float(nash_eq['joint_profit'])
                },
                'cooperative_equilibrium': {
                    'individual_profit': float(coop_eq['individual_profit']),
                    'joint_profit': float(coop_eq['joint_profit'])
                }
            },
            'individual_seed_results': []
        }
        
        # Add individual seed results
        for result in results['individual_results']:
            seed_data = {
                'seed': result['seed'],
                'individual_profit': float(result['summary']['final_individual_profit']),
                'joint_profit': float(result['summary']['final_joint_profit']),
                'nash_ratio': float(result['summary']['nash_ratio_individual']),
                'converged': result['summary']['converged']
            }
            output_data['individual_seed_results'].append(seed_data)
        
        # Save JSON results
        with open(args.json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 JSON results saved to: {args.json}")
        
        # Save CSV results
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write summary
            writer.writerow(['Metric', 'Value', 'Target', 'Pass'])
            writer.writerow(['Individual Profit Mean', f'{individual_mean:.4f}', f'{target_individual:.4f}', individual_pass])
            writer.writerow(['Individual Profit Std', f'{individual_std:.4f}', f'{target_individual_tol:.4f}', ''])
            writer.writerow(['Joint Profit Mean', f'{joint_mean:.4f}', f'{target_joint:.4f}', joint_pass])
            writer.writerow(['Joint Profit Std', f'{joint_std:.4f}', f'{target_joint_tol:.4f}', ''])
            writer.writerow(['Convergence Rate', f'{convergence_rate:.1%}', f'{target_convergence:.1%}', convergence_pass])
            writer.writerow(['Overall Pass', '', '', overall_pass])
            writer.writerow([])
            
            # Write individual seed results
            writer.writerow(['Seed', 'Individual Profit', 'Joint Profit', 'Nash Ratio', 'Converged'])
            for seed_data in output_data['individual_seed_results']:
                writer.writerow([
                    seed_data['seed'],
                    f"{seed_data['individual_profit']:.4f}",
                    f"{seed_data['joint_profit']:.4f}", 
                    f"{seed_data['nash_ratio']:.1%}",
                    seed_data['converged']
                ])
        
        print(f"💾 CSV results saved to: {args.csv}")
        
        # Exit code based on success
        sys.exit(0 if overall_pass else 1)
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Training interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 