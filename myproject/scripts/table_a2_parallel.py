#!/usr/bin/env python3
"""
Parallel Table A.2 replication script for Calvano et al. (2020).

This script runs multiple sessions per seed concurrently using Python's
concurrent.futures to improve performance on multi-core systems.

Usage:
    python -m myproject.scripts.table_a2_parallel --episodes 50000 --n-seeds 10 --n-sessions 4
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from datetime import datetime

# Use relative imports for package modules
from ..train import train_agents


def print_progress(completed: int, total: int, start_time: float, last_result: Dict = None):
    """Print progress information with time estimates."""
    elapsed = time.time() - start_time
    if completed > 0:
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else 0
        eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
    else:
        eta_str = "calculating..."
    
    progress_percent = (completed / total) * 100
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"[{timestamp}] 📊 進捗: {completed}/{total} ({progress_percent:.1f}%) | "
          f"経過時間: {elapsed/60:.1f}m | 残り時間: {eta_str}")
    
    if last_result and last_result.get('success'):
        nash_ratio = last_result.get('nash_ratio_individual', 0) * 100
        individual_profit = last_result.get('final_individual_profit', 0)
        print(f"   最新結果: Individual={individual_profit:.3f}, Nash比={nash_ratio:.1f}%")
    
    # ハートビート（セッション維持）
    print("💓 セッション維持中...", flush=True)
    print()


def run_single_session(
    seed: int,
    session_id: int,
    n_episodes: int,
    iterations_per_episode: int,
    learning_rate: float,
    discount_factor: float,
    epsilon_decay_beta: float,
    memory_length: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single training session with given parameters.
    
    Args:
        seed: Random seed for this session
        session_id: Unique session identifier
        n_episodes: Number of episodes to train
        iterations_per_episode: Iterations per episode
        learning_rate: Q-learning learning rate
        discount_factor: Q-learning discount factor
        epsilon_decay_beta: Epsilon decay parameter
        memory_length: Memory length for state encoding
        verbose: Whether to print progress
        
    Returns:
        Dictionary with session results
    """
    # Use different seed for each session
    session_seed = seed * 1000 + session_id
    
    try:
        # Print session start
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🎯 セッション開始: Seed={seed}, Session={session_id} (総エピソード={n_episodes:,})")
        
        agents, env, history = train_agents(
            n_episodes=n_episodes,
            iterations_per_episode=iterations_per_episode,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_decay_beta=epsilon_decay_beta,
            memory_length=memory_length,
            rng_seed=session_seed,
            verbose=verbose
        )
        
        # Extract final results
        final_individual = history['training_summary']['individual_profit']
        final_joint = history['training_summary']['joint_profit']
        final_epsilon = history['training_summary']['final_epsilon']
        
        # Get equilibrium benchmarks
        nash_eq = env.get_nash_equilibrium()
        coop_eq = env.get_collusive_outcome()
        
        # Print session completion
        nash_ratio = (final_individual / nash_eq['individual_profit']) * 100
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✅ セッション完了: Seed={seed}, Session={session_id} | "
              f"Individual={final_individual:.3f}, Nash比={nash_ratio:.1f}%")
        
        return {
            'seed': seed,
            'session_id': session_id,
            'session_seed': session_seed,
            'final_individual_profit': final_individual,
            'final_joint_profit': final_joint,
            'final_epsilon': final_epsilon,
            'nash_ratio_individual': final_individual / nash_eq['individual_profit'],
            'nash_ratio_joint': final_joint / nash_eq['joint_profit'],
            'cooperative_ratio_individual': final_individual / coop_eq['individual_profit'],
            'cooperative_ratio_joint': final_joint / coop_eq['joint_profit'],
            'nash_equilibrium': {
                'individual_profit': nash_eq['individual_profit'],
                'joint_profit': nash_eq['joint_profit'],
                'price': nash_eq['prices'][0]
            },
            'cooperative_equilibrium': {
                'individual_profit': coop_eq['individual_profit'],
                'joint_profit': coop_eq['joint_profit'],
                'price': coop_eq['prices'][0]
            },
            'training_parameters': {
                'n_episodes': n_episodes,
                'iterations_per_episode': iterations_per_episode,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon_decay_beta': epsilon_decay_beta,
                'memory_length': memory_length
            },
            'beta_info': history['beta_info'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ❌ セッションエラー: Seed={seed}, Session={session_id} | エラー: {str(e)}")
        
        return {
            'seed': seed,
            'session_id': session_id,
            'session_seed': session_seed,
            'success': False,
            'error': str(e),
            'final_individual_profit': None,
            'final_joint_profit': None
        }


def send_completion_notification(results: Dict[str, Any], elapsed_time: float):
    """Send completion notification (can be extended for email, Slack, etc.)"""
    stats = results['aggregated_stats']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    notification = f"""
🎉 フル実験完了通知 🎉

実行時間: {elapsed_time/3600:.1f}時間 ({elapsed_time/60:.0f}分)
完了時刻: {timestamp}

📊 最終結果:
• Individual Profit: {stats['individual_profit_mean']:.4f} ± {stats['individual_profit_std']:.4f}
• Joint Profit: {stats['joint_profit_mean']:.4f} ± {stats['joint_profit_std']:.4f}
• Nash Ratio: {stats['nash_ratio_mean']*100:.1f}% ± {stats['nash_ratio_std']*100:.1f}%

🎯 論文目標値:
• Individual Profit: 0.18 ± 0.03
• Joint Profit: 0.26 ± 0.04

結果ファイル: results/table_a2_parallel.csv
"""
    
    print("=" * 80)
    print(notification)
    print("=" * 80)
    
    # Save notification to file
    with open('results/completion_notification.txt', 'w') as f:
        f.write(notification)
    
    # You can add email/Slack notification here
    # send_email_notification(notification)
    # send_slack_notification(notification)


def run_parallel_experiment(
    n_seeds: int = 10,
    n_sessions_per_seed: int = 4,
    n_episodes: int = 50000,
    iterations_per_episode: int = 25000,
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    epsilon_decay_beta: float = 4e-6,
    memory_length: int = 1,
    max_workers: int = None,
    output_dir: str = "results",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run parallel experiment with multiple sessions per seed.
    
    Args:
        n_seeds: Number of different random seeds
        n_sessions_per_seed: Number of sessions per seed
        n_episodes: Number of episodes per session
        iterations_per_episode: Iterations per episode
        learning_rate: Q-learning learning rate
        discount_factor: Q-learning discount factor
        epsilon_decay_beta: Epsilon decay parameter
        memory_length: Memory length for state encoding
        max_workers: Maximum number of parallel workers
        output_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with aggregated results
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all session parameters
    session_params = []
    for seed in range(n_seeds):
        for session_id in range(n_sessions_per_seed):
            session_params.append({
                'seed': seed,
                'session_id': session_id,
                'n_episodes': n_episodes,
                'iterations_per_episode': iterations_per_episode,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon_decay_beta': epsilon_decay_beta,
                'memory_length': memory_length,
                'verbose': verbose
            })
    
    print(f"🚀 Starting parallel experiment:")
    print(f"   Seeds: {n_seeds}")
    print(f"   Sessions per seed: {n_sessions_per_seed}")
    print(f"   Total sessions: {len(session_params)}")
    print(f"   Episodes per session: {n_episodes:,}")
    print(f"   Iterations per episode: {iterations_per_episode:,}")
    print(f"   Max workers: {max_workers or 'auto'}")
    print()
    
    # Run sessions in parallel
    results = []
    completed = 0
    total_sessions = len(session_params)
    last_result = None
    
    print(f"🚀 並列実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_progress(completed, total_sessions, start_time)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(run_single_session, **params): params 
            for params in session_params
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                last_result = result if result['success'] else last_result
                
                # より頻繁な進捗表示（毎回 or 2回ごと）
                if completed % 2 == 0 or completed == 1 or completed == total_sessions:
                    print_progress(completed, total_sessions, start_time, last_result)
                    
                # 中間結果の保存（進捗の保持）
                if completed % 5 == 0 or completed == total_sessions:
                    save_intermediate_results(results, output_dir, completed, total_sessions)
                
            except Exception as e:
                print(f"❌ Session failed: {e}")
                results.append({
                    'seed': params['seed'],
                    'session_id': params['session_id'],
                    'success': False,
                    'error': str(e)
                })
                completed += 1
    
    # Aggregate results by seed
    seed_results = {}
    for result in results:
        if result['success']:
            seed = result['seed']
            if seed not in seed_results:
                seed_results[seed] = []
            seed_results[seed].append(result)
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        individual_profits = [r['final_individual_profit'] for r in successful_results]
        joint_profits = [r['final_joint_profit'] for r in successful_results]
        nash_ratios = [r['nash_ratio_individual'] for r in successful_results]
        
        aggregated_results = {
            'experiment_info': {
                'n_seeds': n_seeds,
                'n_sessions_per_seed': n_sessions_per_seed,
                'total_sessions': len(session_params),
                'successful_sessions': len(successful_results),
                'execution_time_seconds': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'parameters': {
                'n_episodes': n_episodes,
                'iterations_per_episode': iterations_per_episode,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon_decay_beta': epsilon_decay_beta,
                'memory_length': memory_length
            },
            'aggregated_stats': {
                'individual_profit_mean': float(np.mean(individual_profits)),
                'individual_profit_std': float(np.std(individual_profits)),
                'joint_profit_mean': float(np.mean(joint_profits)),
                'joint_profit_std': float(np.std(joint_profits)),
                'nash_ratio_mean': float(np.mean(nash_ratios)),
                'nash_ratio_std': float(np.std(nash_ratios))
            },
            'seed_results': seed_results,
            'all_sessions': results
        }
    else:
        aggregated_results = {
            'experiment_info': {
                'n_seeds': n_seeds,
                'n_sessions_per_seed': n_sessions_per_seed,
                'total_sessions': len(session_params),
                'successful_sessions': 0,
                'execution_time_seconds': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'error': 'No successful sessions'
        }
    
    # Save results
    output_file = os.path.join(output_dir, f"table_a2_parallel_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Experiment completed in {elapsed_time:.1f} seconds")
    print(f"   Successful sessions: {len(successful_results)}/{len(session_params)}")
    
    if successful_results:
        print(f"   Individual profit: {aggregated_results['aggregated_stats']['individual_profit_mean']:.4f} ± {aggregated_results['aggregated_stats']['individual_profit_std']:.4f}")
        print(f"   Joint profit: {aggregated_results['aggregated_stats']['joint_profit_mean']:.4f} ± {aggregated_results['aggregated_stats']['joint_profit_std']:.4f}")
        print(f"   Nash ratio: {aggregated_results['aggregated_stats']['nash_ratio_mean']:.1%} ± {aggregated_results['aggregated_stats']['nash_ratio_std']:.1%}")
        
        # Send completion notification
        send_completion_notification(aggregated_results, elapsed_time)
    
    print(f"   Results saved to: {output_file}")
    
    return aggregated_results


def save_intermediate_results(results: List[Dict], output_dir: str, completed: int, total: int):
    """中間結果を保存（実験中断時の復旧用）"""
    try:
        # 中間結果ファイル
        intermediate_file = os.path.join(output_dir, f"intermediate_results_{completed}of{total}.json")
        
        with open(intermediate_file, 'w') as f:
            json.dump({
                'completed_sessions': completed,
                'total_sessions': total,
                'completion_percentage': (completed / total) * 100,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
            
        print(f"💾 中間結果保存: {intermediate_file}")
        
    except Exception as e:
        print(f"⚠️  中間結果保存失敗: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parallel Table A.2 replication for Calvano et al. (2020)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=50000,
        help="Number of episodes per session (default: 50000)"
    )
    
    parser.add_argument(
        "--n-seeds", "-s",
        type=int,
        default=10,
        help="Number of random seeds (default: 10)"
    )
    
    parser.add_argument(
        "--n-sessions", "-n",
        type=int,
        default=4,
        help="Number of sessions per seed (default: 4)"
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
        help="Q-learning learning rate (default: 0.15)"
    )
    
    parser.add_argument(
        "--discount-factor", "-g",
        type=float,
        default=0.95,
        help="Q-learning discount factor (default: 0.95)"
    )
    
    parser.add_argument(
        "--epsilon-decay-beta", "-b",
        type=float,
        default=4e-6,
        help="Epsilon decay parameter (default: 4e-6)"
    )
    
    parser.add_argument(
        "--memory-length", "-m",
        type=int,
        default=1,
        help="Memory length for state encoding (default: 1)"
    )
    
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_parallel_experiment(
        n_seeds=args.n_seeds,
        n_sessions_per_seed=args.n_sessions,
        n_episodes=args.episodes,
        iterations_per_episode=args.iterations_per_episode,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_decay_beta=args.epsilon_decay_beta,
        memory_length=args.memory_length,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    return results


if __name__ == "__main__":
    main() 