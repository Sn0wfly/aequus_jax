#!/usr/bin/env python3
"""
Performance Optimization and Validation Script
Tests the MC-CFR learning bug fix, 9-action NLHE system, and bucketing improvements.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import TrainerConfig, PokerTrainer, _cfr_step_pure, _compute_real_cfr_regrets
from poker_bot.core.full_game_engine import play_one_game, batch_play, get_legal_actions_9, apply_action_9
from poker_bot.core.bucketing import compute_info_set_id, validate_bucketing_system, compute_nlhe_info_set_features
from poker_bot.core.mccfr_algorithm import mc_sampling_strategy, accumulate_regrets_fixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mc_cfr_learning_fix():
    """Test that the MC-CFR learning bug is fixed."""
    logger.info("🧪 Testing MC-CFR Learning Bug Fix...")
    
    # Create test configuration
    config = TrainerConfig(
        batch_size=16,
        num_actions=9,
        max_info_sets=1000,
        mc_sampling_rate=0.15
    )
    
    # Initialize test data
    key = jax.random.PRNGKey(42)
    regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
    strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
    
    # Mock LUT data
    lut_keys = jnp.arange(1000, dtype=jnp.int32)
    lut_values = jnp.arange(1000, dtype=jnp.int32)
    lut_table_size = 1000
    
    # Test single CFR step
    start_time = time.time()
    new_regrets, new_strategy = _cfr_step_pure(
        regrets, strategy, key, config, lut_keys, lut_values, lut_table_size
    )
    step_time = time.time() - start_time
    
    # Check that regrets actually changed (learning occurred)
    regret_change = jnp.sum(jnp.abs(new_regrets - regrets))
    strategy_change = jnp.sum(jnp.abs(new_strategy - strategy))
    
    logger.info(f"   Regret change: {regret_change:.6f}")
    logger.info(f"   Strategy change: {strategy_change:.6f}")
    logger.info(f"   Step time: {step_time:.3f}s")
    
    # Validate learning occurred
    learning_occurred = regret_change > 0.001 and strategy_change > 0.001
    logger.info(f"   Learning occurred: {'✅' if learning_occurred else '❌'}")
    
    return learning_occurred

def test_9_action_nlhe_system():
    """Test the 9-action NLHE system implementation."""
    logger.info("🧪 Testing 9-Action NLHE System...")
    
    # Test legal actions computation
    from poker_bot.core.full_game_engine import GameState
    
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
    test_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All 9 actions
    action_results = []
    
    for action in test_actions:
        if legal_actions[action]:
            new_state = apply_action_9(state, action)
            action_results.append({
                'action': action,
                'pot_change': float(new_state.pot - state.pot),
                'stack_change': float(new_state.stacks[2] - state.stacks[2])
            })
    
    logger.info(f"   Action results: {action_results}")
    
    # Test game simulation
    key = jax.random.PRNGKey(42)
    lut_keys = jnp.arange(1000, dtype=jnp.int32)
    lut_values = jnp.arange(1000, dtype=jnp.int32)
    lut_table_size = 1000
    
    payoffs, history, game_data = play_one_game(key, lut_keys, lut_values, lut_table_size, num_actions=9)
    
    logger.info(f"   Game payoffs: {payoffs}")
    logger.info(f"   Game history length: {len(history)}")
    logger.info(f"   Final pot: {game_data['final_pot']}")
    
    # Validate 9-action system
    system_valid = (
        len(legal_actions) == 9 and
        any(legal_actions) and
        len(action_results) > 0 and
        len(history) > 0
    )
    
    logger.info(f"   9-Action system valid: {'✅' if system_valid else '❌'}")
    return system_valid

def test_nlhe_bucketing_system():
    """Test the enhanced NLHE bucketing system."""
    logger.info("🧪 Testing NLHE Bucketing System...")
    
    # Test basic bucketing
    hole_cards = jnp.array([48, 49])  # AA
    community_cards = jnp.full(5, -1)  # Preflop
    player_idx = 0
    pot_size = jnp.array([15.0])
    stack_size = jnp.array([1000.0])
    
    # Test info set ID computation
    info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
    logger.info(f"   Info set ID: {info_set_id}")
    
    # Test NLHE features computation
    features = compute_nlhe_info_set_features(hole_cards, community_cards, player_idx, pot_size, stack_size)
    logger.info(f"   NLHE features: {features}")
    
    # Test different hands
    test_hands = [
        (jnp.array([48, 49]), "AA"),
        (jnp.array([44, 45]), "KK"),
        (jnp.array([48, 44]), "AKs"),
        (jnp.array([48, 43]), "AKo"),
        (jnp.array([23, 0]), "72o")
    ]
    
    buckets = []
    for hole_cards, hand_name in test_hands:
        bucket = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
        buckets.append((hand_name, bucket))
        logger.info(f"   {hand_name}: bucket {bucket}")
    
    # Test postflop bucketing
    flop_cards = jnp.array([0, 1, 2, -1, -1])  # Flop
    flop_bucket = compute_info_set_id(hole_cards, flop_cards, player_idx, pot_size, stack_size)
    logger.info(f"   Postflop bucket: {flop_bucket}")
    
    # Validate bucketing system
    unique_buckets = len(set(bucket for _, bucket in buckets))
    bucketing_valid = (
        unique_buckets >= 4 and  # At least 4 different buckets
        info_set_id >= 0 and info_set_id < 500000 and
        flop_bucket != info_set_id  # Postflop different from preflop
    )
    
    logger.info(f"   Bucketing system valid: {'✅' if bucketing_valid else '❌'}")
    return bucketing_valid

def test_mc_cfr_sampling():
    """Test MC-CFR sampling functionality."""
    logger.info("🧪 Testing MC-CFR Sampling...")
    
    # Test sampling strategy
    regrets = jnp.zeros((1000, 9), dtype=jnp.float32)
    info_set_indices = jnp.arange(1000)
    key = jax.random.PRNGKey(42)
    
    sampling_mask = mc_sampling_strategy(regrets, info_set_indices, key)
    sampling_rate = jnp.mean(sampling_mask)
    
    logger.info(f"   Sampling rate: {sampling_rate:.3f}")
    logger.info(f"   Sampled info sets: {jnp.sum(sampling_mask)}")
    
    # Test regret accumulation
    action_regrets = jnp.ones((1000, 9), dtype=jnp.float32)
    updated_regrets = accumulate_regrets_fixed(regrets, info_set_indices, action_regrets, sampling_mask)
    
    regret_change = jnp.sum(jnp.abs(updated_regrets - regrets))
    logger.info(f"   Regret accumulation change: {regret_change:.6f}")
    
    # Validate MC-CFR sampling
    sampling_valid = (
        0.1 < sampling_rate < 0.2 and  # Around 15% sampling rate
        regret_change > 0  # Regrets were accumulated
    )
    
    logger.info(f"   MC-CFR sampling valid: {'✅' if sampling_valid else '❌'}")
    return sampling_valid

def test_performance_optimization():
    """Test performance optimizations."""
    logger.info("🧪 Testing Performance Optimizations...")
    
    # Test batch processing
    config = TrainerConfig(batch_size=64, num_actions=9, max_info_sets=1000)
    key = jax.random.PRNGKey(42)
    regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
    strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
    
    lut_keys = jnp.arange(1000, dtype=jnp.int32)
    lut_values = jnp.arange(1000, dtype=jnp.int32)
    lut_table_size = 1000
    
    # Warm up JIT
    _cfr_step_pure(regrets, strategy, key, config, lut_keys, lut_values, lut_table_size)
    
    # Performance test
    num_iterations = 10
    start_time = time.time()
    
    for i in range(num_iterations):
        iter_key = jax.random.fold_in(key, i)
        regrets, strategy = _cfr_step_pure(
            regrets, strategy, iter_key, config, lut_keys, lut_values, lut_table_size
        )
    
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    logger.info(f"   Average iteration time: {avg_time:.3f}s")
    logger.info(f"   Iterations per second: {1.0/avg_time:.1f}")
    
    # Test memory efficiency
    memory_estimate = config.max_info_sets * config.num_actions * 4 * 2  # regrets + strategy in bytes
    memory_mb = memory_estimate / (1024 * 1024)
    logger.info(f"   Memory usage: {memory_mb:.1f} MB")
    
    # Validate performance
    performance_valid = avg_time < 1.0  # Less than 1 second per iteration
    
    logger.info(f"   Performance valid: {'✅' if performance_valid else '❌'}")
    return performance_valid

def test_real_cfr_regrets():
    """Test real CFR regret computation."""
    logger.info("🧪 Testing Real CFR Regret Computation...")
    
    # Test regret computation
    hole_cards = jnp.array([48, 49])  # AA
    community_cards = jnp.full(5, -1)  # Preflop
    player_idx = 0
    pot_size = jnp.array([15.0])
    game_payoffs = jnp.array([10.0, -5.0, -2.0, 3.0, -1.0, -5.0])
    strategy = jnp.ones((1000, 9), dtype=jnp.float32) / 9
    num_actions = 9
    
    regrets = _compute_real_cfr_regrets(
        hole_cards, community_cards, player_idx, pot_size, game_payoffs, strategy, num_actions
    )
    
    logger.info(f"   Regrets: {regrets}")
    logger.info(f"   Regret sum: {jnp.sum(regrets):.6f}")
    logger.info(f"   Regret variance: {jnp.var(regrets):.6f}")
    
    # Test different hand strengths
    weak_hand = jnp.array([23, 0])  # 72o
    weak_regrets = _compute_real_cfr_regrets(
        weak_hand, community_cards, player_idx, pot_size, game_payoffs, strategy, num_actions
    )
    
    logger.info(f"   Weak hand regrets: {weak_regrets}")
    
    # Validate regret computation
    regret_valid = (
        jnp.sum(regrets) < 0.1 and  # Should sum to approximately zero
        jnp.var(regrets) > 0 and  # Should have variance
        jnp.any(regrets != 0)  # Should not be all zeros
    )
    
    logger.info(f"   Real CFR regrets valid: {'✅' if regret_valid else '❌'}")
    return regret_valid

def run_comprehensive_validation():
    """Run comprehensive validation of all improvements."""
    logger.info("🚀 Starting Comprehensive Validation...")
    
    results = {}
    
    # Test 1: MC-CFR Learning Bug Fix
    results['mc_cfr_learning'] = test_mc_cfr_learning_fix()
    
    # Test 2: 9-Action NLHE System
    results['nlhe_9_action'] = test_9_action_nlhe_system()
    
    # Test 3: NLHE Bucketing System
    results['nlhe_bucketing'] = test_nlhe_bucketing_system()
    
    # Test 4: MC-CFR Sampling
    results['mc_cfr_sampling'] = test_mc_cfr_sampling()
    
    # Test 5: Performance Optimization
    results['performance'] = test_performance_optimization()
    
    # Test 6: Real CFR Regrets
    results['real_cfr_regrets'] = test_real_cfr_regrets()
    
    # Test 7: Bucketing System Validation
    results['bucketing_validation'] = validate_bucketing_system()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name:20s}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n   Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for production.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review the issues.")
        return False

def main():
    """Main validation function."""
    logger.info("🔧 Aequus JAX Poker AI - Performance Optimization & Validation")
    logger.info("="*70)
    
    # Configure JAX for better performance
    jax.config.update('jax_enable_x64', False)  # Use float32 for speed
    jax.config.update('jax_platform_name', 'cpu')  # Force CPU for consistent testing
    
    try:
        success = run_comprehensive_validation()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())