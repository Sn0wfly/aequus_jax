#!/usr/bin/env python3
"""
Test script to verify complete diversity fix with both enhanced generation and dimensional stacking bucketing.
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import generate_diverse_game_state
from poker_bot.core.bucketing import compute_info_set_id_enhanced
import time

def test_complete_diversity_fix():
    """Test that both enhanced generation and dimensional stacking work together."""
    
    print("🎯 Testing Complete Diversity Fix")
    print("=" * 50)
    print("✅ Enhanced generate_diverse_game_state")
    print("✅ Dimensional Stacking Bucketing")
    print("=" * 50)
    
    # Generate many diverse states
    num_states = 2000
    states = []
    info_sets = set()
    
    print(f"Generating {num_states} diverse game states...")
    
    for i in range(num_states):
        key = jax.random.PRNGKey(i)
        state = generate_diverse_game_state(key)
        states.append(state)
        
        # Create info set key with dimensional stacking
        info_key = compute_info_set_id_enhanced(
            state.hole_cards[0],  # Current player's cards
            state.comm_cards,
            state.cur_player[0],  # Player position
            state.pot,  # Pot size
            state.stacks[0:1],  # Current player's stack
            None,  # Action history
            5000000  # Max info sets (5M)
        )
        info_sets.add(str(info_key))
        
        if i % 200 == 0:
            print(f"Generated {i} states, unique info sets: {len(info_sets)}")
    
    print(f"\n📊 RESULTS:")
    print(f"Total states generated: {num_states}")
    print(f"Unique info sets: {len(info_sets)}")
    print(f"Diversity ratio: {len(info_sets)/num_states:.3f}")
    
    # Analyze diversity by dimension
    all_stacks = []
    all_pots = []
    all_players = []
    all_streets = []
    
    for state in states:
        all_stacks.extend(state.stacks.tolist())
        all_pots.append(state.pot[0].item())
        all_players.append(state.cur_player[0].item())
        all_streets.append(state.street[0].item())
    
    print(f"\n📈 DIMENSIONAL ANALYSIS:")
    print(f"Stack range: {min(all_stacks):.1f} - {max(all_stacks):.1f}")
    print(f"Pot range: {min(all_pots):.1f} - {max(all_pots):.1f}")
    print(f"Player positions: {set(all_players)}")
    print(f"Streets: {set(all_streets)}")
    
    # Calculate expected vs actual diversity
    expected_diversity = num_states * 0.9  # We expect 90% diversity with dimensional stacking
    actual_diversity = len(info_sets)
    
    print(f"\n🎯 DIVERSITY ASSESSMENT:")
    if actual_diversity > expected_diversity:
        print(f"✅ EXCELLENT: {actual_diversity} unique info sets (expected >{expected_diversity:.0f})")
        print(f"   🎉 Dimensional stacking is working perfectly!")
    elif actual_diversity > expected_diversity * 0.7:
        print(f"⚠️  GOOD: {actual_diversity} unique info sets (expected >{expected_diversity:.0f})")
        print(f"   🎯 Dimensional stacking is working well!")
    else:
        print(f"❌ POOR: {actual_diversity} unique info sets (expected >{expected_diversity:.0f})")
        print(f"   💥 Need to fix dimensional stacking!")
    
    # Test collision rate
    collision_rate = 1 - (len(info_sets) / num_states)
    print(f"\n🔍 COLLISION ANALYSIS:")
    print(f"Collision rate: {collision_rate:.3f}")
    if collision_rate < 0.1:
        print(f"✅ EXCELLENT: Low collision rate (<10%)")
    elif collision_rate < 0.3:
        print(f"⚠️  ACCEPTABLE: Moderate collision rate (<30%)")
    else:
        print(f"❌ POOR: High collision rate (>30%)")
    
    return len(info_sets) > expected_diversity * 0.7

if __name__ == "__main__":
    success = test_complete_diversity_fix()
    if success:
        print("\n🎉 Complete diversity fix PASSED! Ready for training.")
        print("🚀 Deploy to vast.ai and train with 500k+ iterations!")
    else:
        print("\n💥 Complete diversity fix FAILED! Need more debugging.") 