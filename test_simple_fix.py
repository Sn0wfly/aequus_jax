#!/usr/bin/env python3
"""
Simple test script to verify JAX boolean conversion fixes.
"""

import jax
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_9_action_system():
    """Test the 9-action system without the complex validation."""
    logger.info("🧪 Testing 9-Action System...")
    
    try:
        from poker_bot.core.full_game_engine import GameState, get_legal_actions_9, apply_action_9
        
        # Create test game state
        state = GameState(
            stacks=jnp.full((6,), 1000.0),
            bets=jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0),
            player_status=jnp.zeros((6,), dtype=jnp.int8),
            hole_cards=jnp.arange(12).reshape(6, 2),
            comm_cards=jnp.full((5,), -1),
            cur_player=jnp.array([2], dtype=jnp.int8),
            street=jnp.array([0], dtype=jnp.int8),
            pot=jnp.array([15.0]),
            deck=jnp.arange(52),
            deck_ptr=jnp.array([12]),
            acted_this_round=jnp.array([0], dtype=jnp.int8),
            key=jax.random.PRNGKey(0),
            action_hist=jnp.zeros((60,), dtype=jnp.int8),
            hist_ptr=jnp.array([0])
        )
        
        # Test legal actions
        legal_actions = get_legal_actions_9(state)
        logger.info(f"   Legal actions: {legal_actions}")
        
        # Test action application
        if legal_actions[0]:  # FOLD
            new_state = apply_action_9(state, 0)
            logger.info(f"   FOLD action applied successfully")
        
        logger.info("   ✅ 9-Action system test passed")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ 9-Action system test failed: {e}")
        return False

def test_mc_cfr_sampling():
    """Test MC-CFR sampling functionality."""
    logger.info("🧪 Testing MC-CFR Sampling...")
    
    try:
        from poker_bot.core.mccfr_algorithm import mc_sampling_strategy, accumulate_regrets_fixed
        
        # Test sampling strategy
        regrets = jnp.zeros((100, 9), dtype=jnp.float32)
        info_set_indices = jnp.arange(100)
        key = jax.random.PRNGKey(42)
        
        sampling_mask = mc_sampling_strategy(regrets, info_set_indices, key)
        sampling_rate = jnp.mean(sampling_mask)
        
        logger.info(f"   Sampling rate: {sampling_rate:.3f}")
        logger.info(f"   Sampled info sets: {jnp.sum(sampling_mask)}")
        
        # Test regret accumulation
        action_regrets = jnp.ones((100, 9), dtype=jnp.float32)
        updated_regrets = accumulate_regrets_fixed(regrets, info_set_indices, action_regrets, sampling_mask)
        
        regret_change = jnp.sum(jnp.abs(updated_regrets - regrets))
        logger.info(f"   Regret accumulation change: {regret_change:.6f}")
        
        logger.info("   ✅ MC-CFR sampling test passed")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ MC-CFR sampling test failed: {e}")
        return False

def test_bucketing_system():
    """Test the bucketing system."""
    logger.info("🧪 Testing Bucketing System...")
    
    try:
        from poker_bot.core.bucketing import compute_info_set_id, validate_bucketing_system
        
        # Test basic bucketing
        hole_cards = jnp.array([48, 49])  # AA
        community_cards = jnp.full(5, -1)  # Preflop
        player_idx = 0
        pot_size = jnp.array([15.0])
        stack_size = jnp.array([1000.0])
        
        # Test info set ID computation
        info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
        logger.info(f"   Info set ID: {info_set_id}")
        
        # Test validation
        validation_result = validate_bucketing_system()
        logger.info(f"   Bucketing validation: {validation_result}")
        
        logger.info("   ✅ Bucketing system test passed")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Bucketing system test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🔧 Simple JAX Fix Validation")
    logger.info("="*50)
    
    # Configure JAX
    jax.config.update('jax_enable_x64', False)
    
    results = {}
    
    # Test 1: 9-Action System
    results['9_action_system'] = test_9_action_system()
    
    # Test 2: MC-CFR Sampling
    results['mc_cfr_sampling'] = test_mc_cfr_sampling()
    
    # Test 3: Bucketing System
    results['bucketing_system'] = test_bucketing_system()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name:20s}: {status}")
    
    logger.info(f"\n   Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! JAX fixes are working.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 