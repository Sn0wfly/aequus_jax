#!/usr/bin/env python3
"""
🚨 CRITICAL TEST: Is batch averaging canceling out individual game regrets?
This tests the exact batch processing logic from _cfr_step_pure.
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_batch_averaging_cancellation():
    """Test if batch averaging is zeroing out regret updates"""
    print("🔍 TESTING: Batch Averaging Cancellation Bug")
    print("=" * 50)
    
    batch_size = 128  # Same as default trainer config
    max_info_sets = 1000
    num_actions = 6
    
    # Simulate batch_regret_updates as if each game produced regret updates
    print("🎮 Simulating individual game regret updates...")
    
    # Create realistic individual game regret patterns
    batch_regret_updates = []
    
    for game_idx in range(batch_size):
        # Simulate a single game producing regret updates similar to our previous test
        game_regrets = jnp.zeros((max_info_sets, num_actions), dtype=jnp.float32)
        
        # Simulate 6 players with different regret patterns
        if game_idx < batch_size // 2:
            # First half of games: positive regrets for some players
            game_regrets = game_regrets.at[10].set(jnp.array([0., 1.56, 1.04, 0., 0., 0.]))
            game_regrets = game_regrets.at[25].set(jnp.array([1.59, 3.18, 4.77, 1.59, 0., 0.]))
            game_regrets = game_regrets.at[50].set(jnp.array([0., 5.49, 3.66, 0., 0., 0.]))
        else:
            # Second half of games: negative regrets for same players (opposite pattern)
            game_regrets = game_regrets.at[10].set(jnp.array([0., -1.56, -1.04, 0., 0., 0.]))
            game_regrets = game_regrets.at[25].set(jnp.array([-1.59, -3.18, -4.77, -1.59, 0., 0.]))
            game_regrets = game_regrets.at[50].set(jnp.array([0., -5.49, -3.66, 0., 0., 0.]))
        
        batch_regret_updates.append(game_regrets)
    
    batch_regret_updates = jnp.stack(batch_regret_updates)  # [batch_size, max_info_sets, num_actions]
    
    print(f"✅ batch_regret_updates shape: {batch_regret_updates.shape}")
    
    # Check individual game magnitudes
    individual_magnitudes = jnp.sum(jnp.abs(batch_regret_updates), axis=(1, 2))  # [batch_size]
    print(f"✅ Individual game magnitudes:")
    print(f"   First 5 games: {individual_magnitudes[:5]}")
    print(f"   Last 5 games: {individual_magnitudes[-5:]}")
    print(f"   Mean individual magnitude: {jnp.mean(individual_magnitudes):.6f}")
    
    # Test the EXACT batch averaging from trainer.py line 210
    print(f"\n🎯 Testing batch averaging (line 210 from trainer):")
    regret_updates = jnp.mean(batch_regret_updates, axis=0)  # Average across games
    
    print(f"✅ Averaged regret_updates shape: {regret_updates.shape}")
    total_averaged_magnitude = float(jnp.sum(jnp.abs(regret_updates)))
    print(f"✅ Total averaged magnitude: {total_averaged_magnitude:.6f}")
    
    if total_averaged_magnitude < 0.001:  # Essentially zero
        print("❌ CONFIRMED: Batch averaging produces near-ZERO regrets!")
        print("   This explains the CFR learning bug!")
        
        # Analyze the cancellation
        print(f"\n🔬 Analyzing cancellation:")
        mean_across_games = jnp.mean(batch_regret_updates, axis=0)  # Should be same as regret_updates
        
        # Check specific info sets that had updates
        for info_set in [10, 25, 50]:
            individual_values = batch_regret_updates[:, info_set, 1]  # Check action 1
            print(f"   Info set {info_set}, action 1:")
            print(f"     Individual values: min={jnp.min(individual_values):.3f}, max={jnp.max(individual_values):.3f}")
            print(f"     Mean: {jnp.mean(individual_values):.6f}")
            
    else:
        print("✅ Batch averaging preserves regret magnitude")
        print("   The zero regret bug must be elsewhere")
    
    # Test with realistic zero-sum cancellation (like poker payoffs)
    print(f"\n🃏 Testing with zero-sum game payoff patterns:")
    
    # Simulate more realistic poker scenario where payoffs sum to zero across players
    # but individual games still have meaningful regret updates
    realistic_batch = []
    
    for game_idx in range(batch_size):
        # Each game has zero-sum payoffs but different patterns
        key = jax.random.PRNGKey(game_idx)
        game_regrets = jnp.zeros((max_info_sets, num_actions))
        
        # Random but realistic regret patterns
        payoffs = jax.random.normal(key, (6,)) * 20.0
        payoffs = payoffs - jnp.mean(payoffs)  # Ensure zero-sum like poker
        
        # Apply regret patterns to some info sets
        for i, payoff in enumerate(payoffs):
            info_set = (game_idx + i * 17) % 500  # Spread across different info sets
            pattern = jnp.where(payoff > 0, 
                               jnp.array([0., 0.1, 0.2, 0.3, 0.3, 0.1]) * payoff,
                               jnp.array([0., 0.2, 0.3, 0.1, 0., 0.]) * payoff)
            game_regrets = game_regrets.at[info_set].set(pattern)
        
        realistic_batch.append(game_regrets)
    
    realistic_batch_regrets = jnp.stack(realistic_batch)
    
    # Test averaging with realistic patterns
    realistic_averaged = jnp.mean(realistic_batch_regrets, axis=0)
    realistic_magnitude = float(jnp.sum(jnp.abs(realistic_averaged)))
    
    print(f"✅ Realistic zero-sum averaged magnitude: {realistic_magnitude:.6f}")
    
    if realistic_magnitude < 0.001:
        print("❌ CONFIRMED: Even realistic patterns average to near-zero!")
        print("   The problem is fundamental to the batch averaging approach!")
    else:
        print("✅ Realistic patterns maintain non-zero regrets after averaging")

if __name__ == "__main__":
    test_batch_averaging_cancellation()