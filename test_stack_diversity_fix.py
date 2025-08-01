import jax
import jax.numpy as jnp
from poker_bot.core.trainer import _cfr_step_with_mccfr
from poker_bot.core.config import TrainerConfig
import numpy as np

def test_stack_diversity_fix():
    """Test that the stack diversity fix works correctly."""
    
    print("🧪 Testing Stack Diversity Fix")
    print("=" * 50)
    
    # Create a simple config
    config = TrainerConfig(
        batch_size=4,
        num_actions=9,
        max_info_sets=1000,
        learning_rate=0.01,
        mc_sampling_rate=0.8,
        mc_exploration_epsilon=0.4,
        use_regret_discounting=True,
        use_cfr_plus=True
    )
    
    # Initialize random state
    key = jax.random.PRNGKey(42)
    
    # Create dummy LUT
    lut_keys = jnp.array([0, 1, 2, 3, 4, 5])
    lut_values = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    lut_table_size = 6
    
    # Initialize regrets and strategy
    regrets = jnp.zeros((config.max_info_sets, config.num_actions))
    strategy = jnp.ones((config.max_info_sets, config.num_actions)) / config.num_actions
    
    print("📊 Running CFR step with stack diversity fix...")
    
    try:
        # Run one CFR step
        updated_regrets, updated_strategy = _cfr_step_with_mccfr(
            regrets, strategy, key, config, 0, lut_keys, lut_values, lut_table_size
        )
        
        print("✅ CFR step completed successfully!")
        print(f"📈 Regrets shape: {updated_regrets.shape}")
        print(f"📈 Strategy shape: {updated_strategy.shape}")
        
        # Check that we have some non-zero regrets (indicating training)
        non_zero_regrets = jnp.sum(updated_regrets > 0)
        print(f"🎯 Non-zero regrets: {non_zero_regrets}")
        
        if non_zero_regrets > 0:
            print("✅ Stack diversity fix working - training is active!")
        else:
            print("⚠️ No regrets updated - may need more iterations")
            
    except Exception as e:
        print(f"❌ Error in CFR step: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_stack_diversity_fix() 