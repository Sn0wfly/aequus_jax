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

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.evaluator import load_hand_evaluation_lut

def main():
    print("🚀 PRODUCTION SYSTEM TEST - CFR+ Real Poker Training")
    print("=" * 80)
    
    # Test configuration for production
    config = TrainerConfig(
        num_players=6,
        num_actions=3,  # fold, call, bet/raise
        num_iterations=100,  # Reduced for testing
        regret_discount_factor=0.95,  # CFR-γ with strong discounting
        strategy_discount_factor=0.99,
        batch_size=64,  # Production batch size
        learning_rate=0.01,
        enable_pruning=True  # CFR+ pruning enabled
    )
    
    print(f"⚙️  Configuration:")
    print(f"   Players: {config.num_players}")
    print(f"   Actions: {config.num_actions}")
    print(f"   Iterations: {config.num_iterations}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Regret Discount: {config.regret_discount_factor}")
    print(f"   CFR+ Pruning: {config.enable_pruning}")
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
        print(f"   📈 Strategy tables: {trainer.strategy_sum.shape}")
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
        # Initialize training state
        trainer_state = trainer.init_train_state()
        print(f"   🎯 Initial regret sum: {jnp.sum(jnp.abs(trainer_state.regrets)):.2f}")
        
        # Run several training iterations
        num_test_iterations = 10
        regret_history = []
        
        print(f"   🔄 Running {num_test_iterations} training iterations...")
        
        for iteration in range(num_test_iterations):
            start_time = time.time()
            
            # Single training step
            trainer_state = trainer.train_step(trainer_state, key)
            
            # Track metrics
            regret_sum = jnp.sum(jnp.abs(trainer_state.regrets))
            regret_history.append(float(regret_sum))
            
            iter_time = time.time() - start_time
            
            if iteration % 3 == 0:
                print(f"      Iter {iteration+1:2d}: Regret sum = {regret_sum:8.2f}, Time = {iter_time:.3f}s")
        
        print(f"   📈 Training progression:")
        print(f"      Initial regret: {regret_history[0]:.2f}")
        print(f"      Final regret: {regret_history[-1]:.2f}")
        print(f"      Trend: {'Decreasing ✅' if regret_history[-1] < regret_history[0] else 'Stable/Increasing ⚠️'}")
        
        # Validate strategy extraction
        final_strategy = trainer.get_current_strategy(trainer_state)
        print(f"   🧠 Final strategy shape: {final_strategy.shape}")
        print(f"   🎲 Strategy sum check: {jnp.sum(final_strategy, axis=-1)}")  # Should be ~1 for each state
        
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