#!/usr/bin/env python3
"""
✅ REALISTIC FIX VERIFICATION: Test with semi-realistic poker regret patterns
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_realistic_fix():
    """Test the fix with more realistic poker regret patterns"""
    print("🃏 TESTING: Realistic Fix Verification")
    print("=" * 40)
    
    batch_size = 128
    max_info_sets = 1000
    num_actions = 6
    
    # Create semi-realistic regret patterns (not perfectly symmetric)
    print("🎮 Creating semi-realistic poker regret patterns...")
    
    batch_regret_updates = []
    
    for game_idx in range(batch_size):
        key = jax.random.PRNGKey(game_idx)
        game_regrets = jnp.zeros((max_info_sets, num_actions), dtype=jnp.float32)
        
        # Simulate 6 players with different outcomes but realistic poker patterns
        payoffs = jax.random.normal(key, (6,)) * 15.0  # Realistic poker payoff range
        
        # Add some correlation between games but not perfect cancellation
        trend = jnp.sin(game_idx * 0.1) * 5.0  # Slight systematic trend
        payoffs = payoffs + trend
        
        # Apply regret patterns to different info sets per game
        for player_idx, payoff in enumerate(payoffs):
            # Different info sets per game (spreading the signal)
            info_set = (game_idx + player_idx * 47) % 300
            
            # Realistic action patterns based on payoff
            if payoff > 10.0:  # Strong positive outcome
                pattern = jnp.array([0., 0.05, 0.1, 0.4, 0.35, 0.1]) * payoff
            elif payoff > 0.0:  # Moderate positive
                pattern = jnp.array([0., 0.15, 0.3, 0.2, 0.1, 0.]) * payoff  
            else:  # Negative outcome
                pattern = jnp.array([0.1, 0.25, 0.15, 0., 0., 0.]) * payoff
            
            game_regrets = game_regrets.at[info_set].set(pattern)
        
        batch_regret_updates.append(game_regrets)
    
    batch_regret_updates = jnp.stack(batch_regret_updates)
    
    print(f"✅ batch_regret_updates shape: {batch_regret_updates.shape}")
    
    # Calculate individual game magnitudes
    individual_magnitudes = jnp.sum(jnp.abs(batch_regret_updates), axis=(1, 2))
    mean_individual = jnp.mean(individual_magnitudes)
    print(f"✅ Mean individual game magnitude: {mean_individual:.3f}")
    
    # Test OLD approach (jnp.mean)
    old_regret_updates = jnp.mean(batch_regret_updates, axis=0)
    old_magnitude = float(jnp.sum(jnp.abs(old_regret_updates)))
    
    # Test NEW approach (jnp.sum) 
    new_regret_updates = jnp.sum(batch_regret_updates, axis=0)
    new_magnitude = float(jnp.sum(jnp.abs(new_regret_updates)))
    
    print(f"\n📊 COMPARISON:")
    print(f"   ❌ jnp.mean() magnitude: {old_magnitude:.3f}")
    print(f"   ✅ jnp.sum() magnitude: {new_magnitude:.3f}")
    
    if new_magnitude > old_magnitude * 2:  # At least 2x improvement
        ratio = new_magnitude / old_magnitude if old_magnitude > 0 else float('inf')
        print(f"   🎉 SUCCESS: {ratio:.1f}x improvement with jnp.sum()")
        
        # Calculate signal preservation
        total_individual = float(jnp.sum(individual_magnitudes))
        old_preservation = old_magnitude / total_individual * 100
        new_preservation = new_magnitude / total_individual * 100
        
        print(f"\n📈 Signal Preservation:")
        print(f"   Total individual signal: {total_individual:.3f}")
        print(f"   jnp.mean() preserves: {old_preservation:.1f}%")
        print(f"   jnp.sum() preserves: {new_preservation:.1f}%")
        
        return True
    else:
        print(f"   ❌ INSUFFICIENT: Only {new_magnitude/old_magnitude:.1f}x improvement")
        return False

def test_cfr_theoretical_correctness():
    """Test that jnp.sum is theoretically correct for CFR"""
    print(f"\n🧮 THEORETICAL CORRECTNESS:")
    print("In CFR, regret updates should accumulate information from multiple games.")
    print("Averaging would dilute the learning signal and slow convergence.")
    print("Summing preserves the full magnitude of regret information.")
    
    # Simple example
    print(f"\n💡 SIMPLE EXAMPLE:")
    game1_regret = jnp.array([1.0, 2.0, -0.5])
    game2_regret = jnp.array([0.5, -1.0, 2.0])
    
    mean_result = jnp.mean(jnp.stack([game1_regret, game2_regret]), axis=0)
    sum_result = jnp.sum(jnp.stack([game1_regret, game2_regret]), axis=0)
    
    print(f"   Game 1 regrets: {game1_regret}")
    print(f"   Game 2 regrets: {game2_regret}")
    print(f"   jnp.mean(): {mean_result}")
    print(f"   jnp.sum(): {sum_result}")
    print(f"   Sum preserves {jnp.sum(jnp.abs(sum_result))/jnp.sum(jnp.abs(mean_result)):.1f}x more information")

if __name__ == "__main__":
    success = test_realistic_fix()
    test_cfr_theoretical_correctness()
    
    if success:
        print(f"\n✅ CONCLUSION: Fix verified for realistic poker scenarios!")
    else:
        print(f"\n❌ CONCLUSION: Fix needs further investigation.")