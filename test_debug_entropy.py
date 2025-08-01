import jax
import jax.numpy as jnp
from poker_bot.core.trainer import PokerTrainer
from poker_bot.core.config import TrainerConfig

def test_debug_entropy():
    """Test the debug entropy prints."""
    
    print("🔍 Testing Debug Entropy Prints")
    print("=" * 40)
    
    # Create a simple config
    config = TrainerConfig(
        batch_size=16,
        num_actions=9,
        max_info_sets=1000,
        learning_rate=0.01,
        mc_sampling_rate=0.8,
        mc_exploration_epsilon=0.4,
        use_regret_discounting=True,
        use_cfr_plus=True
    )
    
    # Create trainer
    trainer = PokerTrainer(config)
    
    print("📊 Initial entropy state:")
    entropy_details = trainer._compute_strategy_entropy_detailed()
    print(f"   Overall entropy: {entropy_details['overall_entropy']:.4f}")
    print(f"   Trained info sets: {entropy_details['trained_info_sets']}")
    print(f"   Trained percentage: {entropy_details['trained_percentage']:.2f}%")
    
    # Run multiple training iterations
    for i in range(1, 6):
        print(f"\n🚀 Running training iteration {i}...")
        
        # Run one training iteration
        results = trainer.train(1, f"test_debug_{i}")
        
        print(f"\n📊 After iteration {i}:")
        entropy_details = trainer._compute_strategy_entropy_detailed()
        print(f"   Overall entropy: {entropy_details['overall_entropy']:.4f}")
        print(f"   Trained info sets: {entropy_details['trained_info_sets']}")
        print(f"   Trained percentage: {entropy_details['trained_percentage']:.2f}%")
    
    return True

if __name__ == "__main__":
    test_debug_entropy() 