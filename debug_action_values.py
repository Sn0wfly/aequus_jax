#!/usr/bin/env python3
"""
Debug script to understand why the bot isn't learning.
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import _evaluate_7card_simple
from poker_bot.core.bucketing import compute_info_set_id_enhanced
from poker_bot.core.starting_hands import classify_starting_hand, evaluate_hand_strength_multi_street
from poker_bot.core.board_analysis import analyze_hand_vs_board

def test_action_values():
    """Test action value calculation to see if it's working correctly."""
    
    # Test different hand strengths
    test_cases = [
        ("Strong hand", 0.8),
        ("Medium hand", 0.5), 
        ("Weak hand", 0.2),
        ("Very weak hand", 0.1)
    ]
    
    pot_size = 100.0
    action_aggressiveness = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=jnp.float32)
    
    print("🎯 ACTION VALUE CALCULATION TEST")
    print("=" * 50)
    
    for hand_type, hand_strength in test_cases:
        strength_modifier = (hand_strength - 0.5) * 2.0
        values = action_aggressiveness * strength_modifier * pot_size
        
        print(f"\n📊 {hand_type} (strength: {hand_strength:.2f})")
        print(f"   Strength modifier: {strength_modifier:.2f}")
        print(f"   Action values: {values}")
        
        # Show which actions are preferred
        best_action_idx = jnp.argmax(values)
        worst_action_idx = jnp.argmin(values)
        
        actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
        print(f"   Best action: {actions[best_action_idx]} (value: {values[best_action_idx]:.2f})")
        print(f"   Worst action: {actions[worst_action_idx]} (value: {values[worst_action_idx]:.2f})")

def test_hand_evaluation():
    """Test hand evaluation to see if it's working correctly."""
    
    print("\n🎯 HAND EVALUATION TEST")
    print("=" * 50)
    
    # Test some sample hands
    test_hands = [
        ("AA", jnp.array([0, 1])),  # Aces
        ("KK", jnp.array([12, 13])),  # Kings  
        ("AK", jnp.array([0, 12])),  # Ace-King
        ("72", jnp.array([5, 18])),  # 7-2 offsuit
        ("T9", jnp.array([8, 9])),  # Ten-Nine
    ]
    
    community_cards = jnp.array([20, 21, 22, -1, -1])  # Flop only
    
    for hand_name, hole_cards in test_hands:
        strength = _evaluate_7card_simple(hole_cards, community_cards, 0)
        print(f"   {hand_name}: strength = {strength:.3f}")

def test_detailed_hand_evaluation():
    """Test each component of hand evaluation separately."""
    
    print("\n🎯 DETAILED HAND EVALUATION TEST")
    print("=" * 50)
    
    # Test hands with different board textures
    test_scenarios = [
        ("Dry board", jnp.array([20, 21, 22, -1, -1])),  # 5-6-7 rainbow
        ("Wet board", jnp.array([0, 4, 8, -1, -1])),     # A-2-3 suited
        ("Paired board", jnp.array([12, 13, 20, -1, -1])), # K-K-5
    ]
    
    test_hands = [
        ("AA", jnp.array([0, 1])),  # Aces
        ("KK", jnp.array([12, 13])),  # Kings  
        ("AK", jnp.array([0, 12])),  # Ace-King
        ("72", jnp.array([5, 18])),  # 7-2 offsuit
    ]
    
    for board_name, community_cards in test_scenarios:
        print(f"\n📊 {board_name}: {community_cards}")
        
        for hand_name, hole_cards in test_hands:
            # Test starting hand classification
            starting_strength = classify_starting_hand(hole_cards)
            
            # Test hand vs board analysis
            board_strength = analyze_hand_vs_board(hole_cards, community_cards)
            
            # Test multi-street evaluation
            multi_street_strength = evaluate_hand_strength_multi_street(hole_cards, community_cards, 0)
            
            # Test final evaluation
            final_strength = _evaluate_7card_simple(hole_cards, community_cards, 0)
            
            print(f"   {hand_name}: board={board_strength:.3f}, final={final_strength:.3f}")

def test_info_set_calculation():
    """Test info set calculation to see if it's working correctly."""
    
    print("\n🎯 INFO SET CALCULATION TEST")
    print("=" * 50)
    
    # Test some sample scenarios
    test_scenarios = [
        ("Preflop AA", jnp.array([0, 1]), jnp.array([-1, -1, -1, -1, -1]), 0, 100, 1000),
        ("Flop AK", jnp.array([0, 12]), jnp.array([20, 21, 22, -1, -1]), 1, 150, 1200),
        ("Turn QQ", jnp.array([10, 11]), jnp.array([30, 31, 32, 33, -1]), 2, 200, 800),
    ]
    
    for scenario_name, hole_cards, community_cards, player_idx, pot, stack in test_scenarios:
        info_set_id = compute_info_set_id_enhanced(
            hole_cards, community_cards, player_idx, 
            jnp.array([pot]), jnp.array([stack]), max_info_sets=5000000
        )
        print(f"   {scenario_name}: info_set_id = {info_set_id}")

def test_realistic_action_values():
    """Test action values with realistic hand strengths from our fixed evaluation."""
    
    print("\n🎯 REALISTIC ACTION VALUES TEST")
    print("=" * 50)
    
    # Test realistic scenarios from our fixed evaluation
    test_scenarios = [
        ("AA on dry board", 0.945),
        ("KK on wet board", 0.760),
        ("AK on paired board", 0.902),
        ("72 on wet board", 0.522),
        ("72 on paired board", 0.522),
    ]
    
    pot_size = 100.0
    action_aggressiveness = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=jnp.float32)
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    for scenario_name, hand_strength in test_scenarios:
        strength_modifier = (hand_strength - 0.5) * 2.0
        values = action_aggressiveness * strength_modifier * pot_size
        
        # Add weak hand penalty (simulating the fix)
        weak_hand_penalty = jnp.where(
            hand_strength < 0.6,  # Changed threshold
            jnp.array([0.0, 0.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0, -400.0], dtype=jnp.float32),  # Increased penalties
            jnp.zeros(9, dtype=jnp.float32)
        )
        values = values + weak_hand_penalty
        
        best_action_idx = jnp.argmax(values)
        worst_action_idx = jnp.argmin(values)
        
        print(f"\n📊 {scenario_name} (strength: {hand_strength:.3f})")
        print(f"   Strength modifier: {strength_modifier:.3f}")
        print(f"   Best action: {actions[best_action_idx]} (value: {values[best_action_idx]:.2f})")
        print(f"   Worst action: {actions[worst_action_idx]} (value: {values[worst_action_idx]:.2f})")
        
        # Show top 3 actions
        top_indices = jnp.argsort(values)[-3:][::-1]
        print(f"   Top 3 actions:")
        for i, idx in enumerate(top_indices):
            print(f"     {i+1}. {actions[idx]}: {values[idx]:.2f}")
        
        # Show if weak hands now prefer folding
        if hand_strength < 0.6:
            fold_value = values[0]  # FOLD
            allin_value = values[8]  # ALL_IN
            print(f"   Weak hand check: FOLD={fold_value:.2f}, ALL_IN={allin_value:.2f}")
            if fold_value > allin_value:
                print(f"   ✅ Weak hand correctly prefers FOLD over ALL_IN")
            else:
                print(f"   ❌ Weak hand still prefers ALL_IN over FOLD")

if __name__ == "__main__":
    test_action_values()
    test_hand_evaluation() 
    test_detailed_hand_evaluation()
    test_realistic_action_values()
    test_info_set_calculation()