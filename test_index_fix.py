#!/usr/bin/env python3
"""
Test script to verify the index fix works.
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import TrainerConfig, _update_regrets_for_game_pure
from poker_bot.core.mccfr_algorithm import accumulate_regrets_fixed

def test_index_fix():
    """Test that large indices are handled correctly"""
    print("🧪 Testing index fix...")
    
    # Create config with larger max_info_sets
    config = TrainerConfig()
    print(f"  max_info_sets: {config.max_info_sets}")
    
    # Create regrets table with larger size
    regrets = jnp.zeros((config.max_info_sets, 9))
    strategy = jnp.ones((config.max_info_sets, 9)) / 9
    
    # Test with the large indices we saw in the diagnostic
    info_set_indices = jnp.array([438027, 438327, 440627, 440927, 443227, 443527])
    action_regrets = jnp.array([
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    sampling_mask = jnp.array([True, True, True, True, True, True])
    
    print(f"  info_set_indices: {info_set_indices}")
    print(f"  regrets shape: {regrets.shape}")
    print(f"  All indices < max_info_sets: {jnp.all(info_set_indices < config.max_info_sets)}")
    
    # Test accumulate_regrets_fixed directly
    updated_regrets = accumulate_regrets_fixed(
        regrets, info_set_indices, action_regrets, sampling_mask
    )
    
    # Check if regrets were actually accumulated
    regret_change = jnp.sum(jnp.abs(updated_regrets - regrets))
    print(f"  Regret change: {regret_change}")
    
    # Check specific indices
    for i, idx in enumerate(info_set_indices):
        if idx < config.max_info_sets:
            regret_sum = jnp.sum(updated_regrets[idx])
            print(f"  Index {idx}: regret_sum = {regret_sum}")
    
    print(f"  ✅ Index fix test: {'PASS' if regret_change > 0 else 'FAIL'}")

def test_update_regrets_for_game():
    """Test the complete update function with large indices"""
    print("\n🧪 Testing complete update function...")
    
    config = TrainerConfig()
    regrets = jnp.zeros((config.max_info_sets, 9))
    strategy = jnp.ones((config.max_info_sets, 9)) / 9
    
    # Create game results with large indices
    game_results = {
        'hole_cards': jnp.array([[51, 47], [50, 46], [45, 41], [40, 36], [35, 31], [30, 26]]),
        'final_community': jnp.array([44, 43, 42, 39, 38]),
        'final_pot': jnp.array(100.0),
        'player_stacks': jnp.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
        'player_bets': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    game_payoffs = jnp.array([50.0, -10.0, -10.0, -10.0, -10.0, -10.0])
    rng_key = jax.random.PRNGKey(42)
    
    regret_updates = _update_regrets_for_game_pure(
        regrets, strategy, game_results, game_payoffs, config.num_actions, rng_key
    )
    
    regret_magnitude = jnp.sum(jnp.abs(regret_updates))
    print(f"  Regret updates magnitude: {regret_magnitude}")
    print(f"  ✅ Complete update test: {'PASS' if regret_magnitude > 0 else 'FAIL'}")

def main():
    """Run all tests"""
    print("🚀 Testing Index Fix")
    print("=" * 50)
    
    test_index_fix()
    test_update_regrets_for_game()
    
    print("\n" + "=" * 50)
    print("🎯 Index fix tests completed!")

if __name__ == "__main__":
    main() 