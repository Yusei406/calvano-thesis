#!/usr/bin/env python3
"""
Figure 2b reproduction with Plotly: Interactive grid sensitivity analysis.
"""

import sys
import argparse
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import run_simulation


def run_grid_sensitivity_analysis(grid_sizes=[11, 21], n_seeds=3, episodes=15000, verbose=True):
    """Run grid sensitivity analysis for Figure 2b."""
    if verbose:
        print("📊 Figure 2b: Interactive Grid Sensitivity Analysis")
        print(f"Grid sizes: {grid_sizes}, Seeds per grid: {n_seeds}, Episodes per run: {episodes:,}")
        print()
    
    results = {}
    
    for grid_size in grid_sizes:
        if verbose:
            print(f"🔄 Testing {grid_size}-point grid...")
        
        grid_results = []
        
        for seed in range(1, n_seeds + 1):
            if verbose:
                print(f"  Seed {seed}/{n_seeds}...")
            
            config = {
                'n_episodes': episodes,
                'n_agents': 2,
                'n_actions': grid_size,
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 0.1,
                'epsilon_decay': 0.95,
                'rng_seed': seed + 100,
                'verbose': False
            }
            
            try:
                result = run_simulation(**config)
                grid_results.append({
                    'seed': seed,
                    'converged': result['convergence']['converged'],
                    'individual_profit': result['final_outcomes']['individual_profit'],
                    'joint_profit': result['final_outcomes']['joint_profit'],
                    'runtime': result['simulation_config']['total_runtime_seconds']
                })
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                grid_results.append({
                    'seed': seed,
                    'converged': False,
                    'individual_profit': 0.0,
                    'joint_profit': 0.0,
                    'runtime': 0.0
                })
        
        # Aggregate results
        converged_results = [r for r in grid_results if r['converged']]
        
        if len(converged_results) > 0:
            results[grid_size] = {
                'convergence_rate': len(converged_results) / len(grid_results),
                'individual_profit': np.mean([r['individual_profit'] for r in converged_results]),
                'individual_std': np.std([r['individual_profit'] for r in converged_results]),
                'joint_profit': np.mean([r['joint_profit'] for r in converged_results]),
                'joint_std': np.std([r['joint_profit'] for r in converged_results]),
                'individual_results': grid_results
            }
        else:
            results[grid_size] = {
                'convergence_rate': 0.0,
                'individual_profit': 0.0,
                'individual_std': 0.0,
                'joint_profit': 0.0,
                'joint_std': 0.0,
                'individual_results': grid_results
            }
        
        if verbose:
            conv_rate = results[grid_size]['convergence_rate'] * 100
            ind_profit = results[grid_size]['individual_profit']
            print(f"  Results: {conv_rate:.0f}% convergence, Individual: {ind_profit:.3f}")
    
    return results


def plot_figure2b_plotly(results, output_file="figs/figure2b_plotly.html"):
    """Create interactive Figure 2b plot using Plotly."""
    # Extract data
    grid_sizes = sorted(results.keys())
    convergence_rates = [results[g]['convergence_rate'] * 100 for g in grid_sizes]
    individual_profits = [results[g]['individual_profit'] for g in grid_sizes]
    individual_stds = [results[g]['individual_std'] for g in grid_sizes]
    joint_profits = [results[g]['joint_profit'] for g in grid_sizes]
    joint_stds = [results[g]['joint_std'] for g in grid_sizes]
    
    # Paper targets
    paper_individual = 0.18
    paper_joint = 0.26
    paper_convergence = 90.0
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Convergence Rate by Grid Size', 
                       'Individual Profit by Grid Size',
                       'Joint Profit by Grid Size'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Convergence Rate
    fig.add_trace(
        go.Bar(
            x=grid_sizes,
            y=convergence_rates,
            name='Convergence Rate',
            marker_color='steelblue',
            opacity=0.7,
            text=[f'{y:.0f}%' for y in convergence_rates],
            textposition='outside',
            hovertemplate='Grid Size: %{x}<br>Convergence Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add paper target line for convergence
    fig.add_hline(y=paper_convergence, line_dash="dash", line_color="red", 
                  annotation_text=f"Paper target ({paper_convergence}%)",
                  row=1, col=1)
    
    # Plot 2: Individual Profit
    fig.add_trace(
        go.Scatter(
            x=grid_sizes,
            y=individual_profits,
            error_y=dict(type='data', array=individual_stds, visible=True),
            mode='markers+lines',
            name='Individual Profit',
            marker=dict(size=12, color='darkgreen'),
            line=dict(width=3),
            hovertemplate='Grid Size: %{x}<br>Individual Profit: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add paper target line for individual profit
    fig.add_hline(y=paper_individual, line_dash="dash", line_color="red", 
                  annotation_text=f"Paper target ({paper_individual})",
                  row=1, col=2)
    
    # Plot 3: Joint Profit
    fig.add_trace(
        go.Scatter(
            x=grid_sizes,
            y=joint_profits,
            error_y=dict(type='data', array=joint_stds, visible=True),
            mode='markers+lines',
            name='Joint Profit',
            marker=dict(size=12, color='purple'),
            line=dict(width=3),
            hovertemplate='Grid Size: %{x}<br>Joint Profit: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>'
        ),
        row=1, col=3
    )
    
    # Add paper target line for joint profit
    fig.add_hline(y=paper_joint, line_dash="dash", line_color="red", 
                  annotation_text=f"Paper target ({paper_joint})",
                  row=1, col=3)
    
    # Update layout
    fig.update_layout(
        title_text="Figure 2b: Interactive Grid Sensitivity Analysis - Calvano et al. (2020) Replication",
        title_x=0.5,
        showlegend=False,
        height=500,
        width=1200,
        font=dict(size=12)
    )
    
    # Update x-axes
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="Grid Size (price points)", row=1, col=col)
    
    # Update y-axes
    fig.update_yaxes(title_text="Convergence Rate (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Individual Profit", row=1, col=2)
    fig.update_yaxes(title_text="Joint Profit", row=1, col=3)
    
    # Save plot
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    
    # Also save as PNG for static use
    png_file = output_file.replace('.html', '.png')
    try:
        fig.write_image(png_file, width=1200, height=500, scale=2)
        print(f"📊 Static Figure 2b saved as: {png_file}")
    except Exception as e:
        print(f"⚠️  Could not save PNG (install kaleido): {e}")
    
    print(f"📊 Interactive Figure 2b saved as: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate Figure 2b with Plotly')
    parser.add_argument('--grids', nargs='+', type=int, default=[11, 21], 
                       help='Grid sizes to test (default: 11 21)')
    parser.add_argument('--seeds', type=int, default=3, help='Seeds per grid (default: 3)')
    parser.add_argument('--episodes', type=int, default=15000, help='Episodes per run (default: 15000)')
    parser.add_argument('--output', type=str, default='figs/figure2b_plotly.html', 
                       help='Output HTML file (default: figs/figure2b_plotly.html)')
    parser.add_argument('--no-show', action='store_true', help='Skip opening browser')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not args.quiet:
        print("📊 Figure 2b: Interactive Grid Sensitivity Analysis")
        print("=" * 60)
    
    # Run analysis
    results = run_grid_sensitivity_analysis(
        grid_sizes=args.grids,
        n_seeds=args.seeds,
        episodes=args.episodes,
        verbose=not args.quiet
    )
    
    # Generate plot
    fig = plot_figure2b_plotly(results, args.output)
    
    if not args.no_show and not args.quiet:
        try:
            fig.show()
        except Exception:
            print("⚠️  Could not open browser automatically")
    
    total_time = time.time() - start_time
    
    if not args.quiet:
        print(f"\n✅ Interactive Figure 2b generation complete! Runtime: {total_time/60:.1f} minutes")
        for grid_size in sorted(results.keys()):
            r = results[grid_size]
            print(f"  {grid_size}-point: {r['convergence_rate']*100:.0f}% convergence, {r['individual_profit']:.3f} profit")


if __name__ == '__main__':
    main()
