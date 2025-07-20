#!/usr/bin/env python3
"""
Test script to debug why strategy is not differentiating despite regret accumulation.
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import TrainerConfig, _regret_matching_pure

def test_strategy_differentiation():
    """Test why strategy remains uniform despite regret accumulation"""
    print("🔍 Testing strategy differentiation...")
    
    config = TrainerConfig()
    print(f"  strategy_threshold: {config.strategy_threshold}")
    print(f"  num_actions: {config.num_actions}")
    
    # Create regrets table with some variation
    regrets = jnp.zeros((1000, 9))
    
    # Add some regret variation to a few info sets
    regrets = regrets.at[100, 0].set(0.1)  # Info set 100, action 0
    regrets = regrets.at[100, 1].set(0.2)  # Info set 100, action 1
    regrets = regrets.at[100, 2].set(0.3)  # Info set 100, action 2
    
    regrets = regrets.at[200, 3].set(0.15)  # Info set 200, action 3
    regrets = regrets.at[200, 4].set(0.25)  # Info set 200, action 4
    regrets = regrets.at[200, 5].set(0.35)  # Info set 200, action 5
    
    print(f"  regrets shape: {regrets.shape}")
    print(f"  regrets magnitude: min={jnp.min(regrets):.6f}, max={jnp.max(regrets):.6f}")
    print(f"  non-zero regrets: {jnp.count_nonzero(regrets)}")
    
    # Test regret matching
    strategy = _regret_matching_pure(regrets, config)
    
    print(f"  strategy shape: {strategy.shape}")
    print(f"  strategy magnitude: min={jnp.min(strategy):.6f}, max={jnp.max(strategy):.6f}")
    
    # Check specific info sets
    print(f"  Info set 100 strategy: {strategy[100]}")
    print(f"  Info set 200 strategy: {strategy[200]}")
    print(f"  Info set 0 strategy (should be uniform): {strategy[0]}")
    
    # Check entropy
    log_probs = jnp.log(jnp.clip(strategy, 1e-10, 1.0))
    entropies = -jnp.sum(strategy * log_probs, axis=1)
    mean_entropy = jnp.mean(entropies)
    
    print(f"  Mean strategy entropy: {mean_entropy:.6f}")
    print(f"  Expected uniform entropy: {jnp.log(9):.6f}")
    
    # Check if strategies are properly normalized
    strategy_sums = jnp.sum(strategy, axis=1)
    print(f"  Strategy sums: min={jnp.min(strategy_sums):.6f}, max={jnp.max(strategy_sums):.6f}")
    
    # Check how many info sets have non-uniform strategies
    uniform_strategy = jnp.ones(9) / 9
    is_uniform = jnp.allclose(strategy, uniform_strategy, atol=1e-6, rtol=1e-6)
    uniform_count = jnp.sum(is_uniform)
    
    print(f"  Uniform strategies: {uniform_count}/{len(strategy)} ({uniform_count/len(strategy)*100:.1f}%)")
    
    return strategy

def test_realistic_regrets():
    """Test with realistic regret magnitudes from training"""
    print("\n🧪 Testing with realistic regret magnitudes...")
    
    config = TrainerConfig()
    
    # Create regrets with realistic magnitudes (like 0.2102 total)
    regrets = jnp.zeros((1000, 9))
    
    # Simulate realistic regret distribution
    # Most info sets have small regrets, some have larger ones
    for i in range(100):
        if i % 10 == 0:  # Every 10th info set has some variation
            regrets = regrets.at[i, 0].set(0.001)  # Small regret
            regrets = regrets.at[i, 1].set(0.002)  # Medium regret
            regrets = regrets.at[i, 2].set(0.003)  # Large regret
    
    print(f"  Realistic regrets magnitude: {jnp.sum(jnp.abs(regrets)):.6f}")
    
    # Test strategy computation
    strategy = _regret_matching_pure(regrets, config)
    
    # Check entropy distribution
    log_probs = jnp.log(jnp.clip(strategy, 1e-10, 1.0))
    entropies = -jnp.sum(strategy * log_probs, axis=1)
    
    print(f"  Entropy stats: min={jnp.min(entropies):.6f}, max={jnp.max(entropies):.6f}, mean={jnp.mean(entropies):.6f}")
    print(f"  Expected uniform entropy: {jnp.log(9):.6f}")
    
    # Check if we have any differentiated strategies
    uniform_entropy = jnp.log(9)
    differentiated = entropies < uniform_entropy - 0.01
    differentiated_count = jnp.sum(differentiated)
    
    print(f"  Differentiated strategies: {differentiated_count}/{len(strategy)} ({differentiated_count/len(strategy)*100:.1f}%)")
    
    return strategy

if __name__ == "__main__":
    print("🔍 STRATEGY DIFFERENTIATION DEBUG TEST")
    print("=" * 50)
    
    strategy1 = test_strategy_differentiation()
    strategy2 = test_realistic_regrets()
    
    print("\n✅ Strategy debug test completed!") 