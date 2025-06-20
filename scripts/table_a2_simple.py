#!/usr/bin/env python3
"""
Simplified Table A.2 replication script for testing.
"""

import argparse
import numpy as np
import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import multi_seed_training

def main():
    parser = argparse.ArgumentParser(description='Simple Calvano Test')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--n-seeds', type=int, default=2)
    
    args = parser.parse_args()
    
    print("🎯 Simple Calvano Test")
    print(f"Episodes: {args.episodes}, Seeds: {args.n_seeds}")
    
    # Run training
    try:
        results = multi_seed_training(
            n_seeds=args.n_seeds,
            n_episodes=args.episodes,
            iterations_per_episode=500
        )
        
        # Extract basic results
        stats = results['summary_stats']
        print(f"✅ Individual: {stats['individual_profit']['mean']:.4f}")
        print(f"✅ Joint: {stats['joint_profit']['mean']:.4f}")
        print("🎉 Test successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
