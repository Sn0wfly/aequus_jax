#!/usr/bin/env python3
"""
Debug script to investigate why strong hands are folding
"""

import jax
import jax.numpy as jnp
from poker_bot.core.board_analysis import analyze_hand_vs_board
from poker_bot.core.starting_hands import classify_starting_hand, evaluate_hand_strength_multi_street

def debug_strong_hand_folding():
    """Debug why strong hands are folding"""
    print("🔍 DEBUGGING STRONG HANDS FOLDING")
    print("=" * 50)
    
    # Test case from the play output
    # Hand strength: 0.850, Cards: [ 6 12], Community: [ 8  0 10 11  3]
    hole_cards = jnp.array([6, 12])
    community_cards = jnp.array([8, 0, 10, 11, 3])
    
    print(f"🎯 Test Case:")
    print(f"   Hole cards: {hole_cards}")
    print(f"   Community: {community_cards}")
    print(f"   Reported strength: 0.850")
    
    # Step 1: Check hand evaluation
    print("\n📊 STEP 1: HAND EVALUATION")
    print("-" * 30)
    
    # Check starting hand classification
    starting_hand = classify_starting_hand(hole_cards)
    print(f"   Starting hand: {starting_hand}")
    
    # Check board analysis
    hand_strength = analyze_hand_vs_board(hole_cards, community_cards)
    print(f"   Board analysis strength: {hand_strength:.3f}")
    
    # Check multi-street evaluation
    multi_street_strength = evaluate_hand_strength_multi_street(hole_cards, community_cards, 0)
    print(f"   Multi-street strength: {multi_street_strength:.3f}")
    
    # Step 2: Simulate action value calculation
    print("\n📊 STEP 2: ACTION VALUE SIMULATION")
    print("-" * 30)
    
    # Simulate the action value calculation from trainer.py
    pot_size = 100.0
    action_aggressiveness = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=jnp.float32)
    
    # Test with the reported strength (0.850)
    reported_strength = 0.850
    strength_modifier = (reported_strength - 0.5) * 2.0
    print(f"   Reported strength: {reported_strength}")
    print(f"   Strength modifier: {strength_modifier:.3f}")
    
    # Calculate base values
    base_values = action_aggressiveness * strength_modifier * pot_size
    print(f"   Base values: {base_values}")
    
    # Calculate weak hand penalty (should be zero for strong hands)
    weak_hand_penalty = jnp.where(
        reported_strength < 0.6,
        jnp.array([0.0, 0.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0, -400.0], dtype=jnp.float32),
        jnp.zeros(9, dtype=jnp.float32)
    )
    print(f"   Weak hand penalty: {weak_hand_penalty}")
    
    # Final values
    final_values = base_values + weak_hand_penalty
    print(f"   Final values: {final_values}")
    
    # Check if FOLD is preferred
    fold_value = final_values[0]
    max_value = jnp.max(final_values)
    fold_preferred = fold_value == max_value
    
    print(f"   FOLD value: {fold_value:.2f}")
    print(f"   Max value: {max_value:.2f}")
    print(f"   FOLD preferred: {fold_preferred}")
    
    # Step 3: Check with actual calculated strength
    print("\n📊 STEP 3: WITH ACTUAL CALCULATED STRENGTH")
    print("-" * 30)
    
    actual_strength = float(hand_strength)
    strength_modifier_actual = (actual_strength - 0.5) * 2.0
    print(f"   Actual strength: {actual_strength:.3f}")
    print(f"   Strength modifier: {strength_modifier_actual:.3f}")
    
    # Calculate base values
    base_values_actual = action_aggressiveness * strength_modifier_actual * pot_size
    print(f"   Base values: {base_values_actual}")
    
    # Calculate weak hand penalty
    weak_hand_penalty_actual = jnp.where(
        actual_strength < 0.6,
        jnp.array([0.0, 0.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0, -400.0], dtype=jnp.float32),
        jnp.zeros(9, dtype=jnp.float32)
    )
    print(f"   Weak hand penalty: {weak_hand_penalty_actual}")
    
    # Final values
    final_values_actual = base_values_actual + weak_hand_penalty_actual
    print(f"   Final values: {final_values_actual}")
    
    # Check if FOLD is preferred
    fold_value_actual = final_values_actual[0]
    max_value_actual = jnp.max(final_values_actual)
    fold_preferred_actual = fold_value_actual == max_value_actual
    
    print(f"   FOLD value: {fold_value_actual:.2f}")
    print(f"   Max value: {max_value_actual:.2f}")
    print(f"   FOLD preferred: {fold_preferred_actual}")

def debug_action_value_calculation():
    """Debug the action value calculation specifically"""
    print("\n🔍 DEBUGGING ACTION VALUE CALCULATION")
    print("=" * 50)
    
    # Test different hand strengths
    test_strengths = [0.3, 0.5, 0.7, 0.85, 0.95]
    
    for strength in test_strengths:
        print(f"\n📊 Testing hand strength: {strength}")
        print("-" * 30)
        
        # Simulate action value calculation
        pot_size = 100.0
        action_aggressiveness = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=jnp.float32)
        
        # Calculate strength modifier
        strength_modifier = (strength - 0.5) * 2.0
        print(f"   Strength modifier: {strength_modifier:.3f}")
        
        # Calculate base values
        base_values = action_aggressiveness * strength_modifier * pot_size
        print(f"   Base values: {base_values}")
        
        # Calculate weak hand penalty
        weak_hand_penalty = jnp.where(
            strength < 0.6,
            jnp.array([0.0, 0.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0, -400.0], dtype=jnp.float32),
            jnp.zeros(9, dtype=jnp.float32)
        )
        print(f"   Weak hand penalty: {weak_hand_penalty}")
        
        # Final values
        final_values = base_values + weak_hand_penalty
        print(f"   Final values: {final_values}")
        
        # Check if FOLD is preferred
        fold_value = final_values[0]
        max_value = jnp.max(final_values)
        fold_preferred = fold_value == max_value
        
        print(f"   FOLD value: {fold_value:.2f}")
        print(f"   Max value: {max_value:.2f}")
        print(f"   FOLD preferred: {fold_preferred}")

if __name__ == "__main__":
    debug_strong_hand_folding()
    debug_action_value_calculation() 