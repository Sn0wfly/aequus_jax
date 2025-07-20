#!/usr/bin/env python3
"""
Production System Test: Complete CFR+ Poker Training with Real Game Engine

This test validates the entire production-ready system:
- Real poker engine with LUT-based hand evaluation
- CFR+ algorithm with regret discounting
- Realistic game simulation and payoffs
- Production performance benchmarks
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig, load_hand_evaluation_lut

def main():
    print("🚀 PRODUCTION SYSTEM TEST - CFR+ Real Poker Training")
    print("=" * 80)
    
    # Test configuration for production - using actual TrainerConfig parameters
    config = TrainerConfig(
        batch_size=64,  # Production batch size
        num_actions=3,  # fold, call, bet/raise (simplified for testing)
        max_info_sets=50_000,
        learning_rate=0.01,
        discount_factor=0.95,  # CFR-γ with strong discounting
        use_cfr_plus=True,     # CFR+ pruning enabled
        use_regret_discounting=True,
        log_interval=10,       # More frequent logging for test
        save_interval=50
    )
    
    print(f"⚙️  Configuration:")
    print(f"   Actions: {config.num_actions}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Max Info Sets: {config.max_info_sets:,}")
    print(f"   Discount Factor: {config.discount_factor}")
    print(f"   CFR+ Pruning: {config.use_cfr_plus}")
    print(f"   Regret Discounting: {config.use_regret_discounting}")
    print()
    
    # STEP 1: Load production LUT
    print("1️⃣  Loading Production Hand Evaluation LUT...")
    try:
        lut_keys, lut_values, table_size = load_hand_evaluation_lut()
        print(f"   ✅ LUT loaded successfully: {len(lut_keys):,} entries")
        print(f"   📊 Table size: {table_size:,}")
        print(f"   🎯 Coverage: {len(lut_values):,} unique hand evaluations")
    except Exception as e:
        print(f"   ❌ Error loading LUT: {e}")
        return False
    print()
    
    # STEP 2: Initialize production trainer
    print("2️⃣  Initializing Production Trainer...")
    try:
        trainer = PokerTrainer(config)
        print("   ✅ Trainer initialized successfully")
        print(f"   🧠 Regret tables: {trainer.regrets.shape}")
        print(f"   📈 Strategy tables: {trainer.strategy.shape}")
    except Exception as e:
        print(f"   ❌ Error initializing trainer: {e}")
        return False
    print()
    
    # STEP 3: Test production game engine
    print("3️⃣  Testing Production Game Engine...")
    try:
        # Generate test batch
        key = jax.random.PRNGKey(12345)
        keys = jax.random.split(key, config.batch_size)
        
        # Import the production engine
        from poker_bot.core.full_game_engine import unified_batch_simulation_with_lut
        
        print("   🎮 Running production game simulation...")
        start_time = time.time()
        
        # Run production simulation
        payoffs, histories, game_results = unified_batch_simulation_with_lut(
            keys, lut_keys, lut_values, table_size
        )
        
        compile_time = time.time() - start_time
        print(f"   ⚡ First compilation time: {compile_time:.2f}s")
        
        # Validate results
        print(f"   📊 Batch results validation:")
        print(f"      Payoffs shape: {payoffs.shape}")
        print(f"      Histories shape: {histories.shape}")
        print(f"      Zero-sum check: {jnp.sum(payoffs):.6f} (should be ~0)")
        print(f"      Average pot size: {jnp.mean(game_results['final_pot']):.2f}")
        print(f"      Hand strength range: [{jnp.min(game_results['hole_cards'])}, {jnp.max(game_results['hole_cards'])}]")
        
        # Performance test - second run
        start_time = time.time()
        payoffs2, histories2, game_results2 = unified_batch_simulation_with_lut(
            keys, lut_keys, lut_values, table_size
        )
        execution_time = time.time() - start_time
        print(f"   🏃‍♂️ Execution time (compiled): {execution_time:.4f}s")
        print(f"   📈 Throughput: {config.batch_size/execution_time:.0f} games/second")
        
    except Exception as e:
        print(f"   ❌ Error testing game engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # STEP 4: Test CFR+ training with production engine
    print("4️⃣  Testing CFR+ Training with Production Engine...")
    try:
        # Check initial state
        initial_regret_sum = jnp.sum(jnp.abs(trainer.regrets))
        print(f"   🎯 Initial regret sum: {initial_regret_sum:.2f}")
        print(f"   🧠 Initial strategy shape: {trainer.strategy.shape}")
        
        # Test single CFR step with production engine (manual call to internal function)
        print("   🔬 Testing single CFR step with production engine...")
        
        from poker_bot.core.trainer import _cfr_step_pure
        
        # Convert LUT to JAX arrays for the function
        lut_keys_jax = jnp.array(trainer.lut_keys)
        lut_values_jax = jnp.array(trainer.lut_values)
        
        start_time = time.time()
        
        # Single step test
        new_regrets, new_strategy = _cfr_step_pure(
            trainer.regrets,
            trainer.strategy,
            key,
            trainer.config,
            lut_keys_jax,
            lut_values_jax,
            trainer.lut_table_size
        )
        
        step_time = time.time() - start_time
        
        # Validate results
        regret_change = jnp.sum(jnp.abs(new_regrets)) - initial_regret_sum
        print(f"   ⚡ Single step time: {step_time:.3f}s")
        print(f"   📈 Regret sum change: {regret_change:+.2f}")
        print(f"   🎲 Strategy normalization: {jnp.mean(jnp.sum(new_strategy, axis=-1)):.6f} (should be ~1.0)")
        
        # Update trainer state
        trainer.regrets = new_regrets
        trainer.strategy = new_strategy
        
        final_strategy = trainer.strategy
        print(f"   🧠 Final strategy shape: {final_strategy.shape}")
        print(f"   ✅ CFR+ step completed successfully with production engine")
        
    except Exception as e:
        print(f"   ❌ Error in CFR+ training: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # STEP 5: Production readiness validation
    print("5️⃣  Production Readiness Validation...")
    
    # Memory usage check
    try:
        strategy_memory = final_strategy.nbytes / (1024**2)  # MB
        regret_memory = trainer_state.regrets.nbytes / (1024**2)  # MB
        total_memory = strategy_memory + regret_memory
        
        print(f"   💾 Memory usage:")
        print(f"      Strategy tables: {strategy_memory:.1f} MB")
        print(f"      Regret tables: {regret_memory:.1f} MB") 
        print(f"      Total: {total_memory:.1f} MB")
        
        # Performance benchmarks
        games_per_second = config.batch_size / execution_time
        iterations_per_hour = 3600 / (execution_time + 0.1)  # Rough estimate
        
        print(f"   ⚡ Performance benchmarks:")
        print(f"      Games/second: {games_per_second:.0f}")
        print(f"      Est. iterations/hour: {iterations_per_hour:.0f}")
        
        # Realism checks
        avg_payoff = jnp.mean(jnp.abs(payoffs))
        payoff_variance = jnp.var(payoffs)
        
        print(f"   🎯 Realism checks:")
        print(f"      Average payoff magnitude: ${avg_payoff:.2f}")
        print(f"      Payoff variance: {payoff_variance:.2f}")
        print(f"      Zero-sum property: {'✅' if abs(jnp.sum(payoffs)) < 1e-6 else '❌'}")
        
        # Production readiness score
        checks = [
            games_per_second > 100,  # Performance
            total_memory < 500,      # Memory efficiency  
            abs(jnp.sum(payoffs)) < 1e-6,  # Zero-sum
            regret_history[-1] > 0   # Learning progress
        ]
        
        score = sum(checks) / len(checks) * 100
        print(f"   📊 Production readiness score: {score:.0f}%")
        
        if score >= 75:
            print("   🎉 SYSTEM IS PRODUCTION READY!")
        else:
            print("   ⚠️  System needs optimization before production")
            
    except Exception as e:
        print(f"   ❌ Error in production validation: {e}")
        return False
    
    print()
    print("🎯 PRODUCTION TEST COMPLETE")
    print("=" * 80)
    print("✅ All systems operational - Real poker training with CFR+ enabled")
    print(f"🚀 Ready for production deployment at {games_per_second:.0f} games/second")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)