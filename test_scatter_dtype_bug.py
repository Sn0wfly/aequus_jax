#!/usr/bin/env python3
"""
🚨 CRITICAL BUG DIAGNOSIS: Scatter operation dtype mismatch
This tests the exact scatter operation causing zero regret updates.
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_scatter_dtype_bug():
    """Test the exact scatter operation that's failing in CFR"""
    print("🔍 TESTING: Scatter operation dtype mismatch bug")
    print("=" * 55)
    
    # Simulate the exact same scenario as in _update_regrets_for_game_pure
    max_info_sets = 1000
    num_actions = 6
    
    # Create regret_updates array (same as in trainer)
    regret_updates = jnp.zeros((max_info_sets, num_actions), dtype=jnp.float32)
    print(f"✅ regret_updates: shape={regret_updates.shape}, dtype={regret_updates.dtype}")
    
    # Simulate info_set_indices from compute_info_set_id (6 players)
    info_set_indices = jnp.array([10, 25, 50, 75, 100, 150], dtype=jnp.int32)
    print(f"✅ info_set_indices: shape={info_set_indices.shape}, dtype={info_set_indices.dtype}")
    print(f"   values: {info_set_indices}")
    
    # Simulate all_action_regrets from the heuristic pattern computation
    # This comes from payoffs * pattern arrays
    payoffs = jnp.array([5.0, -10.0, 15.0, -8.0, 12.0, -14.0], dtype=jnp.float32)
    pattern = jnp.array([0.1, 0.2, 0.3, 0.1, 0.0, 0.0], dtype=jnp.float32)  # Example pattern
    
    # Create the exact same computation as in the trainer
    all_action_regrets = jnp.outer(payoffs, pattern)  # [6, 6] 
    print(f"✅ all_action_regrets: shape={all_action_regrets.shape}, dtype={all_action_regrets.dtype}")
    print(f"   sample values: {all_action_regrets[0]}")
    print(f"   magnitude: min={jnp.min(all_action_regrets):.6f}, max={jnp.max(all_action_regrets):.6f}")
    
    # Test the EXACT scatter operation that's failing
    print("\n🎯 Testing scatter operation...")
    try:
        # This is the exact line from trainer.py:160
        updated_regrets = regret_updates.at[info_set_indices].add(all_action_regrets)
        
        print(f"✅ Scatter operation completed")
        print(f"   Result shape: {updated_regrets.shape}")
        print(f"   Result dtype: {updated_regrets.dtype}")
        
        # Check if any updates actually happened
        diff = updated_regrets - regret_updates
        total_change = float(jnp.sum(jnp.abs(diff)))
        print(f"   Total change magnitude: {total_change:.6f}")
        
        if total_change == 0.0:
            print("❌ CONFIRMED: Scatter operation produced ZERO changes")
            print("   This proves the regret update bug!")
        else:
            print(f"✅ Updates applied successfully: {total_change:.6f}")
            
    except Exception as e:
        print(f"❌ Scatter operation failed: {e}")
        return
    
    # Test with dtype promotion issue
    print("\n🔬 Testing dtype promotion...")
    float64_regrets = all_action_regrets.astype(jnp.float64)
    print(f"   float64_regrets dtype: {float64_regrets.dtype}")
    
    try:
        updated_float64 = regret_updates.at[info_set_indices].add(float64_regrets)
        diff_64 = updated_float64 - regret_updates 
        total_change_64 = float(jnp.sum(jnp.abs(diff_64)))
        print(f"   float64 scatter change: {total_change_64:.6f}")
        
        if total_change != total_change_64:
            print("❌ DTYPE MISMATCH DETECTED: Different results with float64!")
            
    except Exception as e:
        print(f"❌ float64 scatter failed: {e}")

if __name__ == "__main__":
    test_scatter_dtype_bug()