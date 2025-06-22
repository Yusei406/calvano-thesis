#!/usr/bin/env python3
"""
Mathematical validation tests for Calvano implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import DemandEnvironment


def test_nash_cooperative_ordering():
    """Test that Nash price < Cooperative price (economic theory)"""
    env = DemandEnvironment()
    
    nash_eq = env.get_nash_equilibrium()
    coop_eq = env.get_collusive_outcome()
    
    nash_price = nash_eq['prices'][0]
    coop_price = coop_eq['prices'][0]
    
    print(f"Nash price: {nash_price:.4f}")
    print(f"Cooperative price: {coop_price:.4f}")
    
    # Critical assertion: Nash < Cooperative
    assert nash_price < coop_price, f"Nash price ({nash_price:.4f}) should be < Cooperative price ({coop_price:.4f})"
    
    print("✅ Mathematical validation passed: Nash < Cooperative")


if __name__ == "__main__":
    test_nash_cooperative_ordering()
