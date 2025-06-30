#!/bin/bash

# Mid-scale experiment script for Calvano thesis
# Runs 1000 episodes × 5 seeds for ~80 minutes

echo "🚀 Starting mid-scale experiment (1000 episodes × 5 seeds)"
echo "Expected runtime: ~80 minutes"
echo ""

# Create output directories
mkdir -p results figs

# Run the parallel experiment with mid-scale parameters
python -m myproject.scripts.table_a2_parallel \
    --episodes 1000 \
    --n-seeds 5 \
    --n-sessions 1 \
    --max-workers 4 \
    --output-dir results

echo ""
echo "✅ Mid-scale experiment completed!"
echo "📊 Results saved to results/"
echo "📈 Figures saved to figs/" 