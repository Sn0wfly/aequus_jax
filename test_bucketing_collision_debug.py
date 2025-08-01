import jax
import jax.numpy as jnp
from poker_bot.core.trainer import PokerTrainer
from poker_bot.core.config import TrainerConfig

def test_bucketing_collision_debug():
    """Test bucketing collision detection."""
    
    print("🔍 Testing Bucketing Collision Debug")
    print("=" * 50)
    
    # Create config with smaller batch size for faster testing
    config = TrainerConfig(
        batch_size=64,
        num_actions=9,
        max_info_sets=10000,
        learning_rate=0.01,
        mc_sampling_rate=0.8,
        mc_exploration_epsilon=0.4,
        use_regret_discounting=True,
        use_cfr_plus=True
    )
    
    # Create trainer
    trainer = PokerTrainer(config)
    
    print("🚀 Running training iterations to check bucketing collisions...")
    print("   (Debug prints will appear every 1000 iterations)")
    
    # Run multiple iterations to see the bucketing debug
    for i in range(1, 6):
        print(f"\n--- Iteration {i} ---")
        results = trainer.train(1000, f"bucketing_debug_{i}")
        
        # Check entropy after each batch
        entropy_details = trainer._compute_strategy_entropy_detailed()
        print(f"📊 After {i*1000} iterations:")
        print(f"   Overall entropy: {entropy_details['overall_entropy']:.4f}")
        print(f"   Trained info sets: {entropy_details['trained_info_sets']}")
        print(f"   Trained percentage: {entropy_details['trained_percentage']:.2f}%")
    
    print("\n✅ Bucketing collision debug test completed!")
    return True

if __name__ == "__main__":
    test_bucketing_collision_debug() 