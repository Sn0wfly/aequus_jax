#!/usr/bin/env python3
"""
Diagnostic script to identify why regret updates are zero.
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import TrainerConfig, _compute_real_cfr_regrets, _update_regrets_for_game_pure, _cfr_step_pure
from poker_bot.core.mccfr_algorithm import mc_sampling_strategy
from poker_bot.core import full_game_engine as game_engine
from poker_bot.core.bucketing import compute_info_set_id

def test_game_engine_output():
    """Test if game engine produces valid payoffs"""
    print("🧪 Testing game engine output...")
    
    # Test single game
    key = jax.random.PRNGKey(42)
    lut_keys = jnp.arange(1000)
    lut_values = jnp.arange(1000)
    lut_table_size = 1000
    
    payoffs, histories, game_results = game_engine.unified_batch_simulation_with_lut(
        jnp.array([key]), lut_keys, lut_values, lut_table_size
    )
    
    print(f"  Payoffs shape: {payoffs.shape}")
    print(f"  Payoffs sample: {payoffs[0]}")
    print(f"  Payoffs sum: {jnp.sum(payoffs[0])}")
    print(f"  Payoffs non-zero: {jnp.count_nonzero(payoffs[0])}")
    
    # Check if payoffs are reasonable
    payoff_magnitude = jnp.sum(jnp.abs(payoffs[0]))
    print(f"  Payoff magnitude: {payoff_magnitude}")
    print(f"  ✅ Game engine test: {'PASS' if payoff_magnitude > 0 else 'FAIL'}")

def test_info_set_computation():
    """Test if info set computation works"""
    print("\n🧪 Testing info set computation...")
    
    # Test info set computation
    hole_cards = jnp.array([51, 47])  # AA
    community_cards = jnp.array([50, 46, 45, -1, -1])  # KKQ on flop
    player_idx = 0
    pot_size = jnp.array([100.0])
    
    info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size)
    print(f"  Info set ID: {info_set_id}")
    print(f"  ✅ Info set test: {'PASS' if info_set_id >= 0 else 'FAIL'}")

def test_real_cfr_regrets():
    """Test if real CFR regrets computation works"""
    print("\n🧪 Testing real CFR regrets computation...")
    
    # Test data
    hole_cards = jnp.array([51, 47])  # AA
    community_cards = jnp.array([50, 46, 45, -1, -1])  # KKQ on flop
    player_idx = 0
    pot_size = jnp.array([100.0])
    game_payoffs = jnp.array([50.0, -10.0, -10.0, -10.0, -10.0, -10.0])  # Player 0 wins
    strategy = jnp.ones((1000, 9)) / 9  # Uniform strategy
    num_actions = 9
    
    regrets = _compute_real_cfr_regrets(
        hole_cards, community_cards, player_idx, pot_size, 
        game_payoffs, strategy, num_actions
    )
    
    print(f"  Regrets shape: {regrets.shape}")
    print(f"  Regrets: {regrets}")
    print(f"  Regrets magnitude: {jnp.sum(jnp.abs(regrets))}")
    print(f"  ✅ Real CFR regrets test: {'PASS' if jnp.sum(jnp.abs(regrets)) > 0 else 'FAIL'}")

def test_sampling_mask():
    """Test if sampling mask is working correctly"""
    print("\n🧪 Testing sampling mask...")
    
    regrets = jnp.zeros((1000, 9))
    info_set_indices = jnp.arange(100)
    rng_key = jax.random.PRNGKey(42)
    
    sampling_mask = mc_sampling_strategy(regrets, info_set_indices, rng_key)
    sampled_count = jnp.sum(sampling_mask)
    
    print(f"  Total info sets: {len(info_set_indices)}")
    print(f"  Sampled info sets: {sampled_count}")
    print(f"  Sampling rate: {sampled_count/len(info_set_indices):.1%}")
    print(f"  ✅ Sampling mask test: {'PASS' if sampled_count > 0 else 'FAIL'}")

def test_update_regrets_for_game():
    """Test the complete regret update pipeline"""
    print("\n🧪 Testing complete regret update pipeline...")
    
    # Setup
    regrets = jnp.zeros((1000, 9))
    strategy = jnp.ones((1000, 9)) / 9
    game_results = {
        'hole_cards': jnp.array([[51, 47], [50, 46], [45, 41], [40, 36], [35, 31], [30, 26]]),  # 6 players
        'final_community': jnp.array([44, 43, 42, 39, 38]),  # Community cards
        'final_pot': jnp.array(100.0),
        'player_stacks': jnp.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
        'player_bets': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    game_payoffs = jnp.array([50.0, -10.0, -10.0, -10.0, -10.0, -10.0])  # Player 0 wins
    num_actions = 9
    rng_key = jax.random.PRNGKey(42)
    
    # Test regret update
    regret_updates = _update_regrets_for_game_pure(
        regrets, strategy, game_results, game_payoffs, num_actions, rng_key
    )
    
    regret_magnitude = jnp.sum(jnp.abs(regret_updates))
    print(f"  Regret updates magnitude: {regret_magnitude}")
    print(f"  Regret updates shape: {regret_updates.shape}")
    print(f"  Non-zero regrets: {jnp.count_nonzero(regret_updates)}")
    print(f"  ✅ Complete pipeline test: {'PASS' if regret_magnitude > 0 else 'FAIL'}")

def test_cfr_step_debug():
    """Test CFR step with detailed debugging"""
    print("\n🧪 Testing CFR step with debugging...")
    
    config = TrainerConfig()
    regrets = jnp.zeros((1000, 9))
    strategy = jnp.ones((1000, 9)) / 9
    key = jax.random.PRNGKey(42)
    
    # Create dummy LUT
    lut_keys = jnp.arange(1000)
    lut_values = jnp.arange(1000)
    lut_table_size = 1000
    
    print("  Running CFR step...")
    try:
        new_regrets, new_strategy = _cfr_step_pure(
            regrets, strategy, key, config, lut_keys, lut_values, lut_table_size
        )
        
        regret_change = jnp.sum(jnp.abs(new_regrets - regrets))
        strategy_change = jnp.sum(jnp.abs(new_strategy - strategy))
        
        print(f"  Regret change: {regret_change:.6f}")
        print(f"  Strategy change: {strategy_change:.6f}")
        print(f"  ✅ CFR step test: {'PASS' if regret_change > 0 or strategy_change > 0 else 'FAIL'}")
        
    except Exception as e:
        print(f"  ❌ CFR step test: FAIL - {e}")

def main():
    """Run all diagnostic tests"""
    print("🔍 Zero Learning Diagnostic")
    print("=" * 50)
    
    test_game_engine_output()
    test_info_set_computation()
    test_real_cfr_regrets()
    test_sampling_mask()
    test_update_regrets_for_game()
    test_cfr_step_debug()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostic completed!")

if __name__ == "__main__":
    main() 