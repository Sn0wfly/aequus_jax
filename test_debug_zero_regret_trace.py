#!/usr/bin/env python3
"""
🚨 DEBUG SCRIPT: Trace exactly where zero regret updates are introduced
This will help identify the root cause of the CFR learning bug.
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

# Enable debug printing
jax.config.update('jax_debug_prints', True)

def debug_zero_regret_trace():
    """Run minimal CFR iterations with full debug tracing"""
    print("🔍 DEBUGGING: Zero Regret Updates Investigation")
    print("=" * 60)
    
    # Create minimal trainer config for debugging
    config = TrainerConfig(
        batch_size=2,  # Very small batch for detailed tracing
        num_actions=6,
        max_info_sets=1000,  # Small for faster debugging
        learning_rate=0.01,
        use_cfr_plus=True,
        use_regret_discounting=True,
        discount_factor=0.9995
    )
    
    print(f"✅ Config: batch_size={config.batch_size}, num_actions={config.num_actions}")
    
    try:
        # Initialize trainer
        trainer = PokerTrainer(config)
        print(f"✅ Trainer initialized")
        print(f"   Initial regrets sum: {float(jnp.sum(jnp.abs(trainer.regrets))):.6f}")
        
        # Run just ONE CFR iteration with full debug tracing
        print("\n🎯 Running 1 CFR iteration with debug logging...")
        print("-" * 50)
        
        stats = trainer.train(num_iterations=1, save_path="debug_trace")
        
        print("\n📊 RESULTS:")
        print(f"   Final regret sum: {stats.get('final_regret_sum', 0):.6f}")
        print(f"   Iterations per second: {stats.get('iterations_per_second', 0):.1f}")
        
        # Check if regrets actually changed
        regret_magnitude = float(jnp.sum(jnp.abs(trainer.regrets)))
        if regret_magnitude == 0.0:
            print("❌ CONFIRMED: Regret updates are exactly ZERO")
            print("   This proves the CFR learning bug exists")
        else:
            print(f"✅ Regrets changed: magnitude = {regret_magnitude:.6f}")
            
    except Exception as e:
        print(f"❌ Debug trace failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_zero_regret_trace()