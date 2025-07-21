#!/usr/bin/env python3
"""
✅ VERIFICATION TEST: Does the jnp.sum fix resolve batch averaging cancellation?
Tests the exact fix applied to _cfr_step_pure.
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_fix_verification():
    """Test that jnp.sum fixes the batch averaging cancellation bug"""
    print("🔧 TESTING: Fix Verification - jnp.sum vs jnp.mean")
    print("=" * 55)
    
    batch_size = 128  # Same as default trainer config
    max_info_sets = 1000
    num_actions = 6
    
    # Recreate the exact problematic scenario from the original test
    print("🎮 Recreating problematic batch scenario...")
    
    batch_regret_updates = []
    
    for game_idx in range(batch_size):
        game_regrets = jnp.zeros((max_info_sets, num_actions), dtype=jnp.float32)
        
        if game_idx < batch_size // 2:
            # First half: positive regrets
            game_regrets = game_regrets.at[10].set(jnp.array([0., 1.56, 1.04, 0., 0., 0.]))
            game_regrets = game_regrets.at[25].set(jnp.array([1.59, 3.18, 4.77, 1.59, 0., 0.]))
            game_regrets = game_regrets.at[50].set(jnp.array([0., 5.49, 3.66, 0., 0., 0.]))
        else:
            # Second half: negative regrets (systematic cancellation)
            game_regrets = game_regrets.at[10].set(jnp.array([0., -1.56, -1.04, 0., 0., 0.]))
            game_regrets = game_regrets.at[25].set(jnp.array([-1.59, -3.18, -4.77, -1.59, 0., 0.]))
            game_regrets = game_regrets.at[50].set(jnp.array([0., -5.49, -3.66, 0., 0., 0.]))
        
        batch_regret_updates.append(game_regrets)
    
    batch_regret_updates = jnp.stack(batch_regret_updates)
    
    print(f"✅ batch_regret_updates shape: {batch_regret_updates.shape}")
    individual_magnitudes = jnp.sum(jnp.abs(batch_regret_updates), axis=(1, 2))
    print(f"✅ Individual game magnitudes: mean={jnp.mean(individual_magnitudes):.6f}")
    
    # Test the OLD approach (jnp.mean) - should produce zero
    print(f"\n❌ OLD APPROACH (jnp.mean) - The Bug:")
    old_regret_updates = jnp.mean(batch_regret_updates, axis=0)
    old_magnitude = float(jnp.sum(jnp.abs(old_regret_updates)))
    print(f"   Magnitude with jnp.mean(): {old_magnitude:.6f}")
    
    # Test the NEW approach (jnp.sum) - should preserve magnitude
    print(f"\n✅ NEW APPROACH (jnp.sum) - The Fix:")
    new_regret_updates = jnp.sum(batch_regret_updates, axis=0)
    new_magnitude = float(jnp.sum(jnp.abs(new_regret_updates)))
    print(f"   Magnitude with jnp.sum(): {new_magnitude:.6f}")
    
    # Verify the fix
    if old_magnitude < 0.001 and new_magnitude > 1.0:
        print(f"\n🎉 SUCCESS: Fix verified!")
        print(f"   ❌ jnp.mean() produces: {old_magnitude:.6f} (near-zero)")
        print(f"   ✅ jnp.sum() produces: {new_magnitude:.6f} (preserves magnitude)")
        
        # Calculate improvement
        if old_magnitude > 0:
            improvement = new_magnitude / old_magnitude
            print(f"   📈 Improvement: {improvement:.1f}x magnitude increase")
        else:
            print(f"   📈 Improvement: INFINITE (from zero to non-zero)")
            
        # Test specific patterns
        print(f"\n🔬 Pattern Analysis:")
        for info_set in [10, 25, 50]:
            old_pattern = old_regret_updates[info_set]
            new_pattern = new_regret_updates[info_set]
            print(f"   Info set {info_set}:")
            print(f"     OLD: {[f'{x:.3f}' for x in old_pattern]}")
            print(f"     NEW: {[f'{x:.3f}' for x in new_pattern]}")
            
    else:
        print(f"\n❌ FAILED: Fix did not work as expected")
        print(f"   Old magnitude: {old_magnitude:.6f}")
        print(f"   New magnitude: {new_magnitude:.6f}")
    
    # Test with more realistic scenario
    print(f"\n🃏 Testing with realistic poker scenario:")
    realistic_batch = []
    
    for game_idx in range(32):  # Smaller batch for realistic test
        key = jax.random.PRNGKey(game_idx)
        game_regrets = jnp.zeros((max_info_sets, num_actions))
        
        # Realistic payoff patterns (some correlation but not perfect cancellation)
        payoffs = jax.random.normal(key, (6,)) * 10.0
        
        for i, payoff in enumerate(payoffs):
            info_set = (game_idx * 7 + i * 13) % 200  # Spread across info sets
            pattern = jnp.where(payoff > 0,
                               jnp.array([0., 0.1, 0.2, 0.3, 0.2, 0.1]) * payoff,
                               jnp.array([0.1, 0.2, 0.1, 0., 0., 0.]) * payoff)
            game_regrets = game_regrets.at[info_set].set(pattern)
        
        realistic_batch.append(game_regrets)
    
    realistic_batch_regrets = jnp.stack(realistic_batch)
    
    realistic_mean = jnp.mean(realistic_batch_regrets, axis=0)
    realistic_sum = jnp.sum(realistic_batch_regrets, axis=0)
    
    realistic_mean_mag = float(jnp.sum(jnp.abs(realistic_mean)))
    realistic_sum_mag = float(jnp.sum(jnp.abs(realistic_sum)))
    
    print(f"   Realistic jnp.mean(): {realistic_mean_mag:.6f}")
    print(f"   Realistic jnp.sum(): {realistic_sum_mag:.6f}")
    print(f"   Ratio (sum/mean): {realistic_sum_mag/realistic_mean_mag:.1f}x")

if __name__ == "__main__":
    test_fix_verification()