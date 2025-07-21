#!/usr/bin/env python3
"""
Test script to verify TrainerConfig.from_yaml loads all parameters correctly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from poker_bot.core.trainer import TrainerConfig

def test_config_loading():
    """Test that TrainerConfig loads all parameters from YAML correctly"""
    
    # Test 1: Load from existing YAML file
    print("Testing config loading from training_config.yaml...")
    try:
        config = TrainerConfig.from_yaml('config/training_config.yaml')
        print("✅ Successfully loaded config from YAML")
        
        # Verify all parameters are loaded
        print("\nLoaded configuration values:")
        print(f"  batch_size: {config.batch_size}")
        print(f"  num_actions: {config.num_actions}")
        print(f"  max_info_sets: {config.max_info_sets}")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  regret_floor: {config.regret_floor}")
        print(f"  regret_ceiling: {config.regret_ceiling}")
        print(f"  strategy_threshold: {config.strategy_threshold}")
        print(f"  save_interval: {config.save_interval}")
        print(f"  log_interval: {config.log_interval}")
        print(f"  discount_factor: {config.discount_factor}")
        print(f"  use_cfr_plus: {config.use_cfr_plus}")
        print(f"  use_regret_discounting: {config.use_regret_discounting}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Test 2: Test fallback values with empty/missing YAML
    print("\nTesting fallback values...")
    try:
        # Create empty YAML file for testing
        with open('test_empty.yaml', 'w') as f:
            f.write('# Empty YAML file\n')
        
        config = TrainerConfig.from_yaml('test_empty.yaml')
        
        # Verify fallback values are used
        expected_defaults = TrainerConfig()
        assert config.batch_size == expected_defaults.batch_size
        assert config.learning_rate == expected_defaults.learning_rate
        assert config.use_cfr_plus == expected_defaults.use_cfr_plus
        
        print("✅ Fallback values work correctly")
        
        # Clean up
        os.remove('test_empty.yaml')
        
    except Exception as e:
        print(f"❌ Error testing fallback values: {e}")
        return False
    
    # Test 3: Test partial YAML loading
    print("\nTesting partial YAML loading...")
    try:
        # Create partial YAML file
        with open('test_partial.yaml', 'w') as f:
            f.write("""
batch_size: 512
learning_rate: 0.005
use_cfr_plus: false
""")
        
        config = TrainerConfig.from_yaml('test_partial.yaml')
        
        # Verify specified values are used, others use defaults
        assert config.batch_size == 512
        assert config.learning_rate == 0.005
        assert config.use_cfr_plus == False
        assert config.num_actions == 6  # Should use default
        
        print("✅ Partial YAML loading works correctly")
        
        # Clean up
        os.remove('test_partial.yaml')
        
    except Exception as e:
        print(f"❌ Error testing partial loading: {e}")
        return False
    
    print("\n🎉 All tests passed! TrainerConfig.from_yaml is working correctly.")
    return True

if __name__ == "__main__":
    test_config_loading()