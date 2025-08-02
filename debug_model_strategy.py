#!/usr/bin/env python3
"""
Debug script to check the actual strategy from the trained model
"""

import pickle
import jax.numpy as jnp
from poker_bot.core.bucketing import compute_info_set_id_enhanced

def debug_model_strategy():
    """Debug the actual strategy from the trained model"""
    print("🔍 DEBUGGING MODEL STRATEGY")
    print("=" * 50)
    
    # Load the trained model
    model_path = "models/fixed_learning_test.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"   Strategy shape: {model_data['strategy'].shape}")
        print(f"   Regrets shape: {model_data['regrets'].shape}")
        print(f"   Iteration: {model_data.get('iteration', 'Unknown')}")
        
        # Test the specific case from play output
        # Hand strength: 0.850, Cards: [ 6 12], Community: [ 8  0 10 11  3]
        hole_cards = jnp.array([6, 12])
        community_cards = jnp.array([8, 0, 10, 11, 3])
        player_id = 0
        pot_size = 100.0
        stack_size = 1000.0
        
        # Calculate info set ID
        info_set_id = compute_info_set_id_enhanced(
            hole_cards, community_cards, player_id, pot_size, stack_size, max_info_sets=500000
        )
        
        print(f"\n📊 TEST CASE:")
        print(f"   Hole cards: {hole_cards}")
        print(f"   Community: {community_cards}")
        print(f"   Info set ID: {info_set_id}")
        
        # Get strategy for this info set
        if 0 <= info_set_id < model_data['strategy'].shape[0]:
            strategy = model_data['strategy'][info_set_id]
            regrets = model_data['regrets'][info_set_id]
            
            print(f"\n📊 STRATEGY ANALYSIS:")
            print(f"   Strategy: {strategy}")
            print(f"   Regrets: {regrets}")
            
            # Check if strategy is uniform
            is_uniform = jnp.allclose(strategy, jnp.ones_like(strategy) / len(strategy))
            print(f"   Is uniform: {is_uniform}")
            
            # Find best action
            best_action_idx = jnp.argmax(strategy)
            actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
            print(f"   Best action: {actions[best_action_idx]} (idx: {best_action_idx})")
            print(f"   Best action probability: {strategy[best_action_idx]:.4f}")
            
            # Check if FOLD is the best action
            fold_prob = strategy[0]
            print(f"   FOLD probability: {fold_prob:.4f}")
            print(f"   FOLD is best: {best_action_idx == 0}")
            
        else:
            print(f"❌ Info set ID {info_set_id} out of range [0, {model_data['strategy'].shape[0]})")
        
        # Test a few more info sets to see patterns
        print(f"\n📊 RANDOM SAMPLES:")
        import random
        for i in range(5):
            random_id = random.randint(0, min(1000, model_data['strategy'].shape[0] - 1))
            random_strategy = model_data['strategy'][random_id]
            best_idx = jnp.argmax(random_strategy)
            print(f"   Sample {i+1}: {actions[best_idx]} (prob: {random_strategy[best_idx]:.3f})")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    debug_model_strategy() 