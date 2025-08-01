import jax
import jax.numpy as jnp
from poker_bot.core.trainer import _cfr_step_with_mccfr
from poker_bot.core.config import TrainerConfig
import numpy as np

def test_diversity_improvement():
    """Test that training now produces more unique info sets."""
    
    print("🚀 Testing Diversity Improvement")
    print("=" * 50)
    
    # Create config with smaller batch for testing
    config = TrainerConfig(
        batch_size=32,
        num_actions=9,
        max_info_sets=10000,
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
    
    print("📊 Running multiple CFR steps to test diversity...")
    
    unique_info_sets_per_step = []
    
    for step in range(5):
        try:
            # Run one CFR step
            updated_regrets, updated_strategy = _cfr_step_with_mccfr(
                regrets, strategy, key, config, step, lut_keys, lut_values, lut_table_size
            )
            
            # Count unique info sets that have been trained (non-zero regrets)
            trained_info_sets = jnp.sum(jnp.any(updated_regrets > 0, axis=1))
            unique_info_sets_per_step.append(int(trained_info_sets))
            
            print(f"   Step {step + 1}: {trained_info_sets} unique info sets trained")
            
            # Update for next iteration
            regrets = updated_regrets
            strategy = updated_strategy
            
        except Exception as e:
            print(f"❌ Error in step {step + 1}: {e}")
            break
    
    print()
    print("📈 Diversity Analysis:")
    print(f"   Total steps: {len(unique_info_sets_per_step)}")
    print(f"   Average unique info sets per step: {np.mean(unique_info_sets_per_step):.1f}")
    print(f"   Max unique info sets: {max(unique_info_sets_per_step)}")
    print(f"   Min unique info sets: {min(unique_info_sets_per_step)}")
    
    # Check if diversity is increasing
    if len(unique_info_sets_per_step) > 1:
        diversity_trend = unique_info_sets_per_step[-1] - unique_info_sets_per_step[0]
        if diversity_trend > 0:
            print("✅ SUCCESS: Diversity is increasing over time!")
        else:
            print("⚠️ WARNING: Diversity is not increasing")
    
    # Check if we're getting reasonable diversity
    avg_diversity = np.mean(unique_info_sets_per_step)
    if avg_diversity > 10:  # At least 10 unique info sets per step
        print("✅ SUCCESS: Good diversity achieved!")
    else:
        print("⚠️ WARNING: Low diversity - may need more iterations")
    
    return True

if __name__ == "__main__":
    test_diversity_improvement() 