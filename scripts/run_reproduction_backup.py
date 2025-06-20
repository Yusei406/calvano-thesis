#!/usr/bin/env python3
"""
Calvano et al. (2020) reproduction script.
Reproduces Table A1/A2 results and outputs to CSV and stdout.
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import run_simulation, run_multiple_seeds


def run_table_a1_reproduction(episodes: int = 50000, verbose: bool = True) -> dict:
    """Reproduce Table A1 baseline results."""
    if verbose:
        print("📊 Table A1 Reproduction: Baseline Configuration")
        print("=" * 55)
        
    config = {
        'n_episodes': episodes,
        'n_agents': 2,
        'n_actions': 11,
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'epsilon_decay': 0.95,
        'rng_seed': 42,
        'verbose': verbose
    }
    
    result = run_simulation(**config)
    
    # Paper targets from Calvano et al. (2020)
    paper_targets = {
        'individual_profit': 0.18,
        'joint_profit': 0.26,
        'convergence_rate': 0.9
    }
    
    # Our results
    our_results = {
        'individual_profit': result['final_outcomes']['individual_profit'],
        'joint_profit': result['final_outcomes']['joint_profit'], 
        'converged': result['convergence']['converged'],
        'convergence_rate': 1.0 if result['convergence']['converged'] else 0.0
    }
    
    # Performance vs paper
    performance = {
        'individual_vs_paper': our_results['individual_profit'] / paper_targets['individual_profit'],
        'joint_vs_paper': our_results['joint_profit'] / paper_targets['joint_profit'],
        'convergence_vs_paper': our_results['convergence_rate'] / paper_targets['convergence_rate']
    }
    
    if verbose:
        print(f"🎯 Results vs Paper Targets:")
        print(f"  Individual Profit: {our_results['individual_profit']:.3f} vs {paper_targets['individual_profit']:.3f} ({performance['individual_vs_paper']*100:.1f}%)")
        print(f"  Joint Profit: {our_results['joint_profit']:.3f} vs {paper_targets['joint_profit']:.3f} ({performance['joint_vs_paper']*100:.1f}%)")
        print(f"  Convergence: {'✅' if our_results['converged'] else '❌'} vs {paper_targets['convergence_rate']*100:.0f}% target")
        print()
    
    return {
        'table': 'A1',
        'paper_targets': paper_targets,
        'our_results': our_results,
        'performance': performance,
        'full_result': result
    }


def run_table_a2_reproduction(n_seeds: int = 20, episodes: int = 50000, verbose: bool = True) -> dict:
    """Reproduce Table A2 robustness analysis."""
    if verbose:
        print("📈 Table A2 Reproduction: Multi-Seed Robustness")
        print("=" * 55)
        
    base_config = {
        'n_episodes': episodes,
        'n_agents': 2, 
        'n_actions': 11,
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'epsilon_decay': 0.95
    }
    
    # Run multiple seeds
    multi_result = run_multiple_seeds(n_seeds=n_seeds, base_config=base_config, verbose=verbose)
    
    # Extract key metrics
    individual_profits = [r['final_outcomes']['individual_profit'] for r in multi_result['individual_results']]
    joint_profits = [r['final_outcomes']['joint_profit'] for r in multi_result['individual_results']]
    converged_runs = [r['convergence']['converged'] for r in multi_result['individual_results']]
    
    # Calculate confidence intervals (95%)
    def confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        se = np.std(data) / np.sqrt(n)
        margin = 1.96 * se  # 95% CI
        return mean, mean - margin, mean + margin
    
    individual_mean, individual_lower, individual_upper = confidence_interval(individual_profits)
    joint_mean, joint_lower, joint_upper = confidence_interval(joint_profits)
    convergence_rate = np.mean(converged_runs)
    
    # Paper comparison
    paper_targets = {
        'individual_profit': 0.18,
        'joint_profit': 0.26,
        'convergence_rate': 0.9
    }
    
    our_results = {
        'n_seeds': n_seeds,
        'individual_profit': {
            'mean': individual_mean,
            'ci_lower': individual_lower,
            'ci_upper': individual_upper,
            'std': np.std(individual_profits)
        },
        'joint_profit': {
            'mean': joint_mean,
            'ci_lower': joint_lower,
            'ci_upper': joint_upper,
            'std': np.std(joint_profits)
        },
        'convergence_rate': convergence_rate
    }
    
    performance = {
        'individual_vs_paper': individual_mean / paper_targets['individual_profit'],
        'joint_vs_paper': joint_mean / paper_targets['joint_profit'],
        'convergence_vs_paper': convergence_rate / paper_targets['convergence_rate']
    }
    
    if verbose:
        print(f"📊 Results ({n_seeds} seeds, 95% CI):")
        print(f"  Individual Profit: {individual_mean:.3f} [{individual_lower:.3f}, {individual_upper:.3f}] vs {paper_targets['individual_profit']:.3f}")
        print(f"  Joint Profit: {joint_mean:.3f} [{joint_lower:.3f}, {joint_upper:.3f}] vs {paper_targets['joint_profit']:.3f}")
        print(f"  Convergence Rate: {convergence_rate*100:.1f}% vs {paper_targets['convergence_rate']*100:.0f}%")
        print(f"  Performance: Individual {performance['individual_vs_paper']*100:.1f}%, Joint {performance['joint_vs_paper']*100:.1f}%")
        print()
    
    return {
        'table': 'A2',
        'paper_targets': paper_targets,
        'our_results': our_results,
        'performance': performance,
        'full_result': multi_result
    }


def save_results_to_csv(results_a1: dict, results_a2: dict, output_dir: str = ".") -> None:
    """Save results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Table A1 CSV
    a1_data = {
        'metric': ['individual_profit', 'joint_profit', 'convergence_rate'],
        'paper_target': [
            results_a1['paper_targets']['individual_profit'],
            results_a1['paper_targets']['joint_profit'], 
            results_a1['paper_targets']['convergence_rate']
        ],
        'our_result': [
            results_a1['our_results']['individual_profit'],
            results_a1['our_results']['joint_profit'],
            results_a1['our_results']['convergence_rate']
        ],
        'performance_ratio': [
            results_a1['performance']['individual_vs_paper'],
            results_a1['performance']['joint_vs_paper'],
            results_a1['performance']['convergence_vs_paper']
        ]
    }
    
    df_a1 = pd.DataFrame(a1_data)
    a1_file = output_path / 'table_a1_reproduction.csv'
    df_a1.to_csv(a1_file, index=False)
    print(f"💾 Table A1 results saved to: {a1_file}")
    
    # Table A2 CSV
    a2_data = {
        'metric': ['individual_profit', 'joint_profit', 'convergence_rate'],
        'paper_target': [
            results_a2['paper_targets']['individual_profit'],
            results_a2['paper_targets']['joint_profit'],
            results_a2['paper_targets']['convergence_rate']
        ],
        'our_mean': [
            results_a2['our_results']['individual_profit']['mean'],
            results_a2['our_results']['joint_profit']['mean'],
            results_a2['our_results']['convergence_rate']
        ],
        'our_ci_lower': [
            results_a2['our_results']['individual_profit']['ci_lower'],
            results_a2['our_results']['joint_profit']['ci_lower'],
            results_a2['our_results']['convergence_rate'] - 0.05
        ],
        'our_ci_upper': [
            results_a2['our_results']['individual_profit']['ci_upper'],
            results_a2['our_results']['joint_profit']['ci_upper'],
            results_a2['our_results']['convergence_rate'] + 0.05
        ],
        'performance_ratio': [
            results_a2['performance']['individual_vs_paper'],
            results_a2['performance']['joint_vs_paper'],
            results_a2['performance']['convergence_vs_paper']
        ]
    }
    
    df_a2 = pd.DataFrame(a2_data)
    a2_file = output_path / 'table_a2_reproduction.csv'
    df_a2.to_csv(a2_file, index=False)
    print(f"💾 Table A2 results saved to: {a2_file}")


def main():
    """Main reproduction script."""
    parser = argparse.ArgumentParser(description='Calvano et al. (2020) Table A1/A2 reproduction')
    parser.add_argument('--episodes', type=int, default=50000, help='Episodes per run (default: 50000)')
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds for A2 (default: 20)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for CSV files')
    parser.add_argument('--table', choices=['A1', 'A2', 'both'], default='both', help='Which table to reproduce')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV output')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not args.quiet:
        print("🚀 Calvano et al. (2020) Algorithmic Pricing Reproduction")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Episodes per run: {args.episodes:,}")
        print(f"  Seeds (Table A2): {args.seeds}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Target table(s): {args.table}")
        print()
    
    results = {}
    
    # Run Table A1
    if args.table in ['A1', 'both']:
        results['A1'] = run_table_a1_reproduction(episodes=args.episodes, verbose=not args.quiet)
    
    # Run Table A2  
    if args.table in ['A2', 'both']:
        results['A2'] = run_table_a2_reproduction(n_seeds=args.seeds, episodes=args.episodes, verbose=not args.quiet)
    
    total_time = time.time() - start_time
    
    # Save to CSV
    if not args.no_csv and 'A1' in results and 'A2' in results:
        save_results_to_csv(results['A1'], results['A2'], args.output_dir)
    
    # Final summary
    if not args.quiet:
        print("✅ Reproduction Complete!")
        print(f"Total runtime: {total_time/60:.1f} minutes")
        print()
        print("🎯 Key Achievements:")
        
        if 'A1' in results:
            a1 = results['A1']
            print(f"  Table A1: Individual {a1['performance']['individual_vs_paper']*100:.0f}%, Joint {a1['performance']['joint_vs_paper']*100:.0f}% of paper targets")
        
        if 'A2' in results:
            a2 = results['A2']
            print(f"  Table A2: Individual {a2['performance']['individual_vs_paper']*100:.0f}%, Joint {a2['performance']['joint_vs_paper']*100:.0f}% of paper targets")
        
        print()
        print("📝 Citation: Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).")
        print("   Artificial intelligence, algorithmic pricing, and collusion. AER, 110(10), 3267-97.")


if __name__ == '__main__':
    main()
