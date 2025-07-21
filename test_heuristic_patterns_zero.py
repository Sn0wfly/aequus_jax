#!/usr/bin/env python3
"""
🚨 TEST: Are the heuristic patterns themselves producing zeros?
This tests the actual pattern computation from the trainer.
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_heuristic_pattern_computation():
    """Test the exact heuristic pattern logic from _update_regrets_for_game_pure"""
    print("🔍 TESTING: Heuristic Pattern Computation")
    print("=" * 50)
    
    # Simulate realistic game payoffs (from game engine)
    game_payoffs = jnp.array([5.2, -12.8, 18.3, -7.1, 15.9, -19.5], dtype=jnp.float32)
    print(f"✅ game_payoffs: {game_payoffs}")
    print(f"   magnitude: min={jnp.min(game_payoffs):.1f}, max={jnp.max(game_payoffs):.1f}")
    
    # Simulate hand strength evaluation (from _evaluate_hand_simple_pure)
    hole_cards = jnp.array([
        [12, 25],  # A♠ K♦  - Strong hand
        [8, 9],    # 3♠ 3♦  - Pair  
        [1, 15],   # 2♠ 4♦  - Weak hand
        [20, 33],  # 6♠ 9♦  - Medium
        [44, 45],  # Q♠ Q♦  - Strong pair
        [2, 6]     # 2♦ 2♠  - Low pair
    ], dtype=jnp.int8)
    
    # Compute hand strengths using the same logic as trainer
    def _evaluate_hand_simple_pure(hole_cards: jnp.ndarray) -> jnp.ndarray:
        """Same logic as in trainer.py"""
        ranks = hole_cards // 4
        suits = hole_cards % 4
        
        # Simple scoring: high cards + pair bonus + suited bonus
        rank_value = jnp.sum(ranks) * 100
        pair_bonus = jnp.where(ranks[0] == ranks[1], 2000, 0)
        suited_bonus = jnp.where(suits[0] == suits[1], 500, 0)
        
        return rank_value + pair_bonus + suited_bonus
    
    hand_strengths = jax.vmap(_evaluate_hand_simple_pure)(hole_cards)
    normalized_strengths = hand_strengths / 10000.0
    
    print(f"✅ hand_strengths: {hand_strengths}")
    print(f"✅ normalized_strengths: {normalized_strengths}")
    
    # Test the exact heuristic pattern computation from trainer
    def compute_regret_vector(strength, payoff, num_actions=6):
        """Exact same logic as in trainer.py lines 134-155"""
        if num_actions == 3:  # FOLD, CALL, BET
            return jnp.where(
                strength > 0.7,
                payoff * jnp.array([0.0, 0.1, 0.8]),  # Fuerte: bet
                jnp.where(
                    strength > 0.3,
                    payoff * jnp.array([0.1, 0.3, 0.1]),  # Medio: mixed
                    payoff * jnp.array([0.0, 0.3, 0.0])   # Débil: call
                )
            )
        else:  # num_actions == 6: FOLD, CHECK, CALL, BET, RAISE, ALL_IN
            return jnp.where(
                strength > 0.7,
                payoff * jnp.array([0.0, 0.0, 0.1, 0.5, 0.8, 0.2]),  # Fuerte: bet/raise
                jnp.where(
                    strength > 0.3,
                    payoff * jnp.array([0.1, 0.2, 0.3, 0.1, 0.0, 0.0]),  # Medio: mixed
                    payoff * jnp.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0])   # Débil: fold/check
                )
            )
    
    # Test for num_actions=6 (default in trainer)
    print(f"\n🎯 Testing 6-action heuristic patterns:")
    all_action_regrets = jax.vmap(compute_regret_vector, in_axes=(0, 0, None))(
        normalized_strengths, game_payoffs, 6
    )
    
    print(f"✅ all_action_regrets shape: {all_action_regrets.shape}")
    print(f"✅ all_action_regrets:")
    for i, (strength, payoff, regrets) in enumerate(zip(normalized_strengths, game_payoffs, all_action_regrets)):
        print(f"   Player {i}: strength={strength:.3f}, payoff={payoff:.1f}")
        print(f"             regrets={regrets}")
        regret_mag = float(jnp.sum(jnp.abs(regrets)))
        print(f"             magnitude={regret_mag:.3f}")
        
        if regret_mag == 0.0:
            print(f"             ❌ ZERO regrets detected!")
        else:
            print(f"             ✅ Non-zero regrets")
    
    total_magnitude = float(jnp.sum(jnp.abs(all_action_regrets)))
    print(f"\n📊 TOTAL PATTERN MAGNITUDE: {total_magnitude:.6f}")
    
    if total_magnitude == 0.0:
        print("❌ CONFIRMED: Heuristic patterns produce ZERO regrets")
        print("   This explains the CFR learning bug!")
        
        # Analyze why patterns are zero
        print(f"\n🔬 Analyzing zero patterns:")
        print(f"   Strengths > 0.7: {jnp.sum(normalized_strengths > 0.7)} players")
        print(f"   Strengths > 0.3: {jnp.sum(normalized_strengths > 0.3)} players") 
        print(f"   Strengths <= 0.3: {jnp.sum(normalized_strengths <= 0.3)} players")
        
    else:
        print("✅ Heuristic patterns produce non-zero regrets")
        print("   The zero regret bug must be elsewhere")

if __name__ == "__main__":
    test_heuristic_pattern_computation()