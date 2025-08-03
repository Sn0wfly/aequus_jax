#!/usr/bin/env python3
"""
Simple test to verify the new generate_diverse_game_state function works.
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import generate_diverse_game_state

def test_simple_diversity():
    """Simple test of diverse state generation."""
    print("🧪 Testing Simple Diversity...")
    
    # Generate a few states
    states = []
    for i in range(5):
        key = jax.random.PRNGKey(i)
        state = generate_diverse_game_state(key, num_players=6)
        states.append(state)
        print(f"State {i}: street={int(state.street[0])}, pot={float(state.pot[0]):.1f}, player={int(state.cur_player[0])}")
    
    # Check that they're different
    streets = [int(state.street[0]) for state in states]
    pots = [float(state.pot[0]) for state in states]
    players = [int(state.cur_player[0]) for state in states]
    
    print(f"\n📊 Unique streets: {len(set(streets))}")
    print(f"📊 Unique pots: {len(set([round(p, 0) for p in pots]))}")
    print(f"📊 Unique players: {len(set(players))}")
    
    # Success if we have some diversity
    if len(set(streets)) >= 2 and len(set(players)) >= 3:
        print("✅ SUCCESS: States are diverse!")
        return True
    else:
        print("❌ FAILURE: States lack diversity!")
        return False

if __name__ == "__main__":
    print("🚀 Simple Diversity Test")
    print("=" * 30)
    
    try:
        success = test_simple_diversity()
        if success:
            print("\n🎉 Test passed! The PRNGKey fix is working!")
        else:
            print("\n❌ Test failed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 