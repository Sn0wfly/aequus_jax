#!/usr/bin/env python3
"""
Test suite for the refactored PokerTrainer with LUT parameter loading
"""

import os
import tempfile
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from poker_bot.core.trainer import PokerTrainer, TrainerConfig, load_hand_evaluation_lut

def create_test_lut():
    """Create a test LUT file for testing purposes"""
    test_lut = {
        'keys': np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        'values': np.array([100, 200, 300, 400, 500, 600], dtype=np.int32),
        'table_size': 6
    }
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(test_lut, f)
        return f.name

def test_lut_loading():
    """Test LUT loading functionality"""
    print("Testing LUT loading...")
    
    # Test with valid LUT file
    lut_path = create_test_lut()
    try:
        lut_keys, lut_values, table_size = load_hand_evaluation_lut(lut_path)
        assert lut_keys.shape == (6,)
        assert lut_values.shape == (6,)
        assert table_size == 6
        assert jnp.array_equal(lut_keys, jnp.array([0, 1, 2, 3, 4, 5]))
        print("✅ LUT loading test passed")
    finally:
        os.unlink(lut_path)
    
    # Test with non-existent file (should use fallback)
    lut_keys, lut_values, table_size = load_hand_evaluation_lut("nonexistent.pkl")
    assert lut_keys is not None
    assert lut_values is not None
    assert table_size > 0
    print("✅ LUT fallback test passed")

def test_trainer_initialization():
    """Test PokerTrainer initialization with LUT"""
    print("Testing trainer initialization...")
    
    config = TrainerConfig(
        batch_size=4,
        max_info_sets=1000,
        num_actions=3
    )
    
    # Test without LUT path
    trainer = PokerTrainer(config)
    assert trainer.lut_keys is not None
    assert trainer.lut_values is not None
    assert trainer.lut_table_size > 0
    print("✅ Trainer initialization without LUT path passed")
    
    # Test with LUT path
    lut_path = create_test_lut()
    try:
        trainer_with_lut = PokerTrainer(config, lut_path=lut_path)
        assert trainer_with_lut.lut_table_size == 6
        print("✅ Trainer initialization with LUT path passed")
    finally:
        os.unlink(lut_path)

def test_cfr_step_integration():
    """Test that _cfr_step properly uses LUT parameters"""
    print("Testing CFR step integration...")
    
    config = TrainerConfig(
        batch_size=2,
        max_info_sets=100,
        num_actions=3
    )
    
    trainer = PokerTrainer(config)
    
    # Test that LUT parameters are properly passed to simulation
    key = jax.random.PRNGKey(42)
    regrets = jnp.zeros((100, 3))
    strategy = jnp.ones((100, 3)) / 3
    
    # This should not raise any errors
    updated_regrets, updated_strategy = trainer._cfr_step(regrets, strategy, key)
    
    assert updated_regrets.shape == (100, 3)
    assert updated_strategy.shape == (100, 3)
    assert not jnp.any(jnp.isnan(updated_regrets))
    assert not jnp.any(jnp.isnan(updated_strategy))
    print("✅ CFR step integration test passed")

def test_backward_compatibility():
    """Test that the trainer maintains backward compatibility"""
    print("Testing backward compatibility...")
    
    # Test old-style initialization (should still work)
    config = TrainerConfig()
    trainer = PokerTrainer(config)
    
    # Test basic functionality
    assert trainer.regrets.shape[0] == config.max_info_sets
    assert trainer.strategy.shape[1] == config.num_actions
    
    # Test that we can save and load
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        trainer.save_model(f.name)
        
        # Create new trainer and load
        new_trainer = PokerTrainer(config)
        new_trainer.load_model(f.name)
        
        assert new_trainer.regrets.shape == trainer.regrets.shape
        assert new_trainer.strategy.shape == trainer.strategy.shape
        
        os.unlink(f.name)
    
    print("✅ Backward compatibility test passed")

def test_training_cycle():
    """Test a complete training cycle"""
    print("Testing training cycle...")
    
    config = TrainerConfig(
        batch_size=2,
        max_info_sets=100,
        num_actions=3,
        log_interval=1,
        save_interval=2
    )
    
    trainer = PokerTrainer(config)
    
    # Run a short training cycle
    stats = trainer.train(5, "test_model")
    
    assert stats['iterations_completed'] == 5
    assert 'final_regret_sum' in stats
    assert 'final_strategy_entropy' in stats
    assert stats['final_regret_sum'] >= 0
    assert stats['final_strategy_entropy'] >= 0
    
    print("✅ Training cycle test passed")

def main():
    """Run all tests"""
    print("🧪 Starting PokerTrainer refactoring tests...")
    
    test_lut_loading()
    test_trainer_initialization()
    test_cfr_step_integration()
    test_backward_compatibility()
    test_training_cycle()
    
    print("🎉 All tests passed! The refactored PokerTrainer is working correctly.")

if __name__ == "__main__":
    main()