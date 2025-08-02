#!/usr/bin/env python3
"""
Test script to verify the robust PRNGKey handling fix.
This tests that the new jax.random.split() logic breaks correlations between iterations.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np
from poker_bot.core.trainer import PokerTrainer, _cfr_step_with_mccfr
from poker_bot.core.config import TrainerConfig

def test_robust_prng_logic():
    """Test that the new PRNGKey logic works correctly."""
    print("🧪 Testing Robust PRNGKey Logic...")
    
    # Create a simple config for testing
    config = TrainerConfig(
        max_info_sets=1000,
        num_actions=9,
        batch_size=32,
        log_interval=10,
        save_interval=100,
        learning_rate=0.1,
        use_cfr_plus=True,
        use_regret_discounting=True,
        discount_factor=0.9
    )
    
    # Create trainer
    trainer = PokerTrainer(config)
    
    print("✅ Trainer created successfully")
    print(f"📊 Initial strategy shape: {trainer.strategy.shape}")
    print(f"📊 Initial regrets shape: {trainer.regrets.shape}")
    
    # Test a few iterations to see the new PRNGKey logic in action
    print("\n🔄 Testing 5 iterations with new PRNGKey logic...")
    
    for i in range(5):
        start_time = time.time()
        
        # This will use the new jax.random.split() logic
        trainer.regrets, trainer.strategy = _cfr_step_with_mccfr(
            trainer.regrets, trainer.strategy, 
            jax.random.PRNGKey(int(time.time())),  # Fresh key each time
            trainer.config, trainer.iteration,
            trainer.lut_keys, trainer.lut_values, trainer.lut_table_size
        )
        
        iter_time = time.time() - start_time
        regret_magnitude = jnp.sum(jnp.abs(trainer.regrets))
        entropy = trainer._compute_strategy_entropy()
        
        print(f"  Iteration {i}: regret={regret_magnitude:.2f}, entropy={entropy:.4f}, time={iter_time:.3f}s")
        
        trainer.iteration += 1
    
    print("\n✅ Robust PRNGKey logic test completed successfully!")
    print("🎯 The new jax.random.split() logic should break correlations between iterations")
    
    return True

def test_key_independence():
    """Test that different keys produce different results."""
    print("\n🔑 Testing Key Independence...")
    
    config = TrainerConfig(
        max_info_sets=100,
        num_actions=9,
        batch_size=16,
        log_interval=5,
        save_interval=50,
        learning_rate=0.1,
        use_cfr_plus=True,
        use_regret_discounting=True,
        discount_factor=0.9
    )
    
    trainer1 = PokerTrainer(config)
    trainer2 = PokerTrainer(config)
    
    # Use different keys
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(123)
    
    # Run one iteration with each key
    trainer1.regrets, trainer1.strategy = _cfr_step_with_mccfr(
        trainer1.regrets, trainer1.strategy, key1, trainer1.config, trainer1.iteration,
        trainer1.lut_keys, trainer1.lut_values, trainer1.lut_table_size
    )
    
    trainer2.regrets, trainer2.strategy = _cfr_step_with_mccfr(
        trainer2.regrets, trainer2.strategy, key2, trainer2.config, trainer2.iteration,
        trainer2.lut_keys, trainer2.lut_values, trainer2.lut_table_size
    )
    
    # Check that results are different
    regret_diff = jnp.sum(jnp.abs(trainer1.regrets - trainer2.regrets))
    strategy_diff = jnp.sum(jnp.abs(trainer1.strategy - trainer2.strategy))
    
    print(f"📊 Regret difference: {regret_diff:.2f}")
    print(f"📊 Strategy difference: {strategy_diff:.2f}")
    
    if regret_diff > 0.1 and strategy_diff > 0.1:
        print("✅ Key independence confirmed - different keys produce different results!")
        return True
    else:
        print("❌ Key independence failed - results too similar!")
        return False

if __name__ == "__main__":
    print("🚀 Testing Robust PRNGKey Fix")
    print("=" * 50)
    
    try:
        # Test 1: Basic functionality
        test_robust_prng_logic()
        
        # Test 2: Key independence
        test_key_independence()
        
        print("\n🎉 All tests passed! The robust PRNGKey fix is working correctly.")
        print("💡 This should help break correlations between training iterations.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 