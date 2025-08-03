#!/usr/bin/env python3
"""
Test script to verify that the new generate_diverse_game_state function
generates truly diverse states without correlations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import generate_diverse_game_state

def test_diverse_state_generation():
    """Test that the new function generates diverse states."""
    print("🧪 Testing Diverse State Generation...")
    
    # Generate multiple states with different keys
    states = []
    for i in range(10):
        key = jax.random.PRNGKey(i)
        state = generate_diverse_game_state(key, num_players=6)
        states.append(state)
    
    print(f"✅ Generated {len(states)} diverse states")
    
    # Check diversity in different aspects
    streets = [int(state.street[0]) for state in states]
    pots = [float(state.pot[0]) for state in states]
    current_players = [int(state.cur_player[0]) for state in states]
    stack_scenarios = []
    
    for state in states:
        # Determine stack scenario based on stack values
        stacks = state.stacks
        if jnp.allclose(stacks, jnp.full((6,), 1000.0), atol=100):
            stack_scenarios.append(0)  # Normal
        elif jnp.allclose(stacks, jnp.full((6,), 100.0), atol=50):
            stack_scenarios.append(3)  # All short
        elif jnp.allclose(stacks, jnp.full((6,), 5000.0), atol=500):
            stack_scenarios.append(4)  # All deep
        else:
            stack_scenarios.append(1)  # Mixed
    
    print(f"📊 Streets: {streets}")
    print(f"📊 Pots: {[f'{p:.1f}' for p in pots]}")
    print(f"📊 Current Players: {current_players}")
    print(f"📊 Stack Scenarios: {stack_scenarios}")
    
    # Check diversity
    unique_streets = len(set(streets))
    unique_pots = len(set([round(p, 0) for p in pots]))
    unique_players = len(set(current_players))
    unique_stack_scenarios = len(set(stack_scenarios))
    
    print(f"\n🎯 Diversity Analysis:")
    print(f"  Unique streets: {unique_streets}/4 possible")
    print(f"  Unique pot ranges: {unique_pots}/10 generated")
    print(f"  Unique players: {unique_players}/6 possible")
    print(f"  Unique stack scenarios: {unique_stack_scenarios}/5 possible")
    
    # Check for correlations (this was the original problem)
    print(f"\n🔍 Correlation Analysis:")
    
    # Check if pot size correlates with current player
    pot_player_pairs = list(zip(pots, current_players))
    unique_pairs = len(set(pot_player_pairs))
    print(f"  Unique pot-player combinations: {unique_pairs}/{len(pot_player_pairs)}")
    
    # Check if street correlates with stack scenario
    street_stack_pairs = list(zip(streets, stack_scenarios))
    unique_street_stack = len(set(street_stack_pairs))
    print(f"  Unique street-stack combinations: {unique_street_stack}/{len(street_stack_pairs)}")
    
    # Success criteria
    success = (
        unique_streets >= 3 and  # At least 3 different streets
        unique_pots >= 8 and     # At least 8 different pot sizes
        unique_players >= 4 and  # At least 4 different players
        unique_stack_scenarios >= 3 and  # At least 3 different stack scenarios
        unique_pairs >= 8 and    # At least 8 unique pot-player combinations
        unique_street_stack >= 8  # At least 8 unique street-stack combinations
    )
    
    if success:
        print("\n✅ SUCCESS: States are truly diverse with minimal correlations!")
        print("🎯 The PRNGKey fix is working correctly!")
        return True
    else:
        print("\n❌ FAILURE: States show correlations or lack diversity!")
        return False

def test_key_independence():
    """Test that different keys produce different states."""
    print("\n🔑 Testing Key Independence...")
    
    # Generate states with same key but different splits
    key = jax.random.PRNGKey(42)
    state1 = generate_diverse_game_state(key, num_players=6)
    
    # Generate state with different key
    key2 = jax.random.PRNGKey(123)
    state2 = generate_diverse_game_state(key2, num_players=6)
    
    # Compare key aspects
    street_diff = abs(state1.street[0] - state2.street[0])
    pot_diff = abs(state1.pot[0] - state2.pot[0])
    player_diff = abs(state1.cur_player[0] - state2.cur_player[0])
    stack_diff = jnp.sum(jnp.abs(state1.stacks - state2.stacks))
    
    print(f"📊 Street difference: {street_diff}")
    print(f"📊 Pot difference: {pot_diff:.1f}")
    print(f"📊 Player difference: {player_diff}")
    print(f"📊 Stack difference: {stack_diff:.1f}")
    
    if street_diff > 0 or pot_diff > 10 or player_diff > 0 or stack_diff > 100:
        print("✅ Key independence confirmed - different keys produce different states!")
        return True
    else:
        print("❌ Key independence failed - states too similar!")
        return False

if __name__ == "__main__":
    print("🚀 Testing Diverse State Generation Fix")
    print("=" * 50)
    
    try:
        # Test 1: Diversity generation
        diversity_success = test_diverse_state_generation()
        
        # Test 2: Key independence
        independence_success = test_key_independence()
        
        if diversity_success and independence_success:
            print("\n🎉 ALL TESTS PASSED! The PRNGKey correlation fix is working!")
            print("💡 This should dramatically increase trained_sets and reduce entropy!")
        else:
            print("\n❌ Some tests failed. The fix may need adjustment.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 