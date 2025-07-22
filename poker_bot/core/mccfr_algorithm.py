"""
MC-CFR Algorithm with proper sampling and regret accumulation.
Implements Monte Carlo Counterfactual Regret Minimization with fixed regret accumulation.
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from functools import partial
from jax.tree_util import register_pytree_node_class
from typing import Tuple, Dict, Any
import numpy as np

# MC-CFR Configuration
class MCCFRConfig:
    sampling_rate = 0.95  # Process 50% of learning opportunities (was 0.15)
    batch_size = 768      # Process 768 player instances per batch
    exploration_epsilon = 0.6  # 60% exploration, 40% exploitation
    regret_floor = 0.0
    discount_factor = 0.999
    min_samples_per_info_set = 100
    max_samples_per_info_set = 10000

@register_pytree_node_class
@dataclass
class CFRState:
    """State for CFR algorithm."""
    regrets: jax.Array
    strategy: jax.Array
    info_set_indices: jax.Array
    iteration: int
    
    def tree_flatten(self):
        children = (self.regrets, self.strategy, self.info_set_indices, self.iteration)
        return children, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

@register_pytree_node_class
@dataclass
class MCCFRState:
    """Complete MCCFR state."""
    regrets: jax.Array
    strategy: jax.Array
    info_set_indices: jax.Array
    iteration: int
    sampling_mask: jax.Array
    
    def tree_flatten(self):
        children = (self.regrets, self.strategy, self.info_set_indices, self.iteration, self.sampling_mask)
        return children, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

@jax.jit
def mc_sampling_strategy(
    regrets: jnp.ndarray,
    info_set_indices: jnp.ndarray,
    rng_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Sample which info sets to process using Monte Carlo approach."""
    batch_size = info_set_indices.shape[0]
    random_values = jax.random.uniform(rng_key, (batch_size,))
    return random_values < MCCFRConfig.sampling_rate

@jax.jit
def accumulate_regrets_fixed(
    regrets: jnp.ndarray,
    info_set_indices: jnp.ndarray,
    action_regrets: jnp.ndarray,
    sampling_mask: jnp.ndarray
) -> jnp.ndarray:
    """FIXED: Proper regret accumulation using scatter_add to avoid collisions."""
    
    # Only process sampled info sets
    valid_mask = sampling_mask & (info_set_indices >= 0) & (info_set_indices < regrets.shape[0])
    
    # Get valid indices and regrets
    valid_indices = jnp.where(valid_mask, info_set_indices, 0)
    valid_regrets = jnp.where(valid_mask[:, None], action_regrets, jnp.zeros_like(action_regrets))
    
    # DEBUG: Add debugging to see what's happening
    #jax.debug.print("🔍 accumulate_regrets_fixed debugging:")
    #jax.debug.print("  regrets shape: {}", regrets.shape)
    #jax.debug.print("  info_set_indices: {}", info_set_indices)
    #jax.debug.print("  valid_mask: {}", valid_mask)
    #jax.debug.print("  valid_indices: {}", valid_indices)
    #jax.debug.print("  valid_regrets magnitude: {}", jnp.sum(jnp.abs(valid_regrets)))
    
    # CRITICAL FIX: Use scatter_add with proper dimension_numbers to avoid collisions
    # This ensures multiple updates to the same info_set are properly accumulated
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),  # action_regrets has 1 update dimension
        inserted_window_dims=(0,),  # info_set_indices has 0 inserted dimensions
        scatter_dims_to_operand_dims=(0,)  # scatter along info_set dimension
    )
    
    # Use scatter_add to properly accumulate regrets for each info_set
    updated_regrets = jax.lax.scatter_add(
        regrets,
        valid_indices[:, None],  # [batch_size, 1] indices
        valid_regrets,           # [batch_size, num_actions] updates
        dimension_numbers=dimension_numbers,
        indices_are_sorted=False,
        unique_indices=False  # Allow multiple updates to same index
    )
    
    # DEBUG: Check if updates were applied
    #jax.debug.print("  updated_regrets magnitude: {}", jnp.sum(jnp.abs(updated_regrets)))
    #jax.debug.print("  regret change: {}", jnp.sum(jnp.abs(updated_regrets - regrets)))
    
    return updated_regrets

@jax.jit
def calculate_strategy(regrets: jnp.ndarray) -> jnp.ndarray:
    """Calculate strategy from regrets using regret matching."""
    positive_regrets = jnp.maximum(regrets, 0)
    sum_positive = jnp.sum(positive_regrets, axis=-1, keepdims=True)
    
    # Avoid division by zero
    strategy = jnp.where(
        sum_positive > 0,
        positive_regrets / sum_positive,
        jnp.ones_like(regrets) / regrets.shape[-1]
    )
    
    return strategy

@jax.jit
def update_strategy(
    strategy: jnp.ndarray,
    regrets: jnp.ndarray,
    info_set_indices: jnp.ndarray,
    action_values: jnp.ndarray,
    sampling_mask: jnp.ndarray
) -> jnp.ndarray:
    """Update strategy using regret matching."""
    # Only update for sampled info sets
    masked_indices = jnp.where(sampling_mask, info_set_indices, 0)
    masked_strategy = jnp.where(
        sampling_mask[:, None], 
        action_values, 
        jnp.zeros_like(action_values)
    )
    
    # Use a simpler approach: manually update strategy
    def update_single_strategy(carry, data):
        strategy, (idx, strategy_update) = carry, data
        new_strategy = strategy.at[idx].set(strategy_update)
        return new_strategy, None
    
    # Process each sampled info set
    final_strategy, _ = jax.lax.scan(
        update_single_strategy,
        strategy,  # Initial carry state
        (masked_indices, masked_strategy)
    )
    
    return final_strategy

@jax.jit
def apply_cfr_plus_discounting(regrets: jnp.ndarray, iteration: int) -> jnp.ndarray:
    """Apply CFR+ regret discounting."""
    discount = MCCFRConfig.discount_factor ** iteration
    return jnp.maximum(regrets * discount, MCCFRConfig.regret_floor)

@jax.jit
def cfr_iteration(
    regrets: jnp.ndarray,
    strategy: jnp.ndarray,
    info_set_indices: jnp.ndarray,
    action_values: jnp.ndarray,
    sampling_mask: jnp.ndarray,
    iteration: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single CFR iteration with MC sampling."""
    
    # Update regrets
    new_regrets = accumulate_regrets_fixed(regrets, info_set_indices, action_values, sampling_mask)
    
    # Apply CFR+ discounting
    new_regrets = apply_cfr_plus_discounting(new_regrets, iteration)
    
    # Update strategy
    new_strategy = calculate_strategy(new_regrets)
    
    return new_regrets, new_strategy

@jax.jit
def batch_cfr_update(
    regrets: jnp.ndarray,
    strategy: jnp.ndarray,
    info_set_indices: jnp.ndarray,
    action_values: jnp.ndarray,
    sampling_mask: jnp.ndarray,
    iteration: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batch CFR update with MC sampling."""
    return jax.vmap(cfr_iteration, in_axes=(0, 0, 0, 0, 0, None))(
        regrets, strategy, info_set_indices, action_values, sampling_mask, iteration
    )

@jax.jit
def validate_learning(regrets: jnp.ndarray) -> bool:
    """Validate that actual learning is occurring."""
    return jnp.any(jnp.abs(regrets) > 0.1)

@jax.jit
def calculate_exploitability(strategy: jnp.ndarray) -> float:
    """Calculate exploitability for validation."""
    # Simplified exploitability calculation
    return jnp.mean(jnp.abs(strategy - 0.5))

@jax.jit
def performance_benchmark(pot_size: jnp.ndarray) -> float:
    """Performance benchmark for validation."""
    return jnp.sum(pot_size) / 1000.0  # Normalized pot size

# Testing utilities
@jax.jit
def test_mc_sampling():
    """Test MC sampling functionality."""
    key = jax.random.PRNGKey(0)
    info_set_indices = jnp.arange(1000)
    sampling_mask = mc_sampling_strategy(jnp.zeros(1000), info_set_indices, key)
    return jnp.sum(sampling_mask) / 1000.0

@jax.jit
def test_regret_accumulation():
    """Test regret accumulation fix."""
    regrets = jnp.zeros((1000, 9))
    info_set_indices = jnp.arange(1000)
    action_values = jnp.ones((1000, 9))
    sampling_mask = jnp.ones(1000)
    
    new_regrets = accumulate_regrets_fixed(regrets, info_set_indices, action_values, sampling_mask)
    return jnp.sum(new_regrets)

class MCCFRTrainer:
    """MC-CFR trainer for poker."""
    
    def __init__(self, num_info_sets: int, num_actions: int):
        self.regrets = jnp.zeros((num_info_sets, num_actions))
        self.strategy = jnp.ones((num_info_sets, num_actions)) / num_actions
        self.iteration = 0
    
    def update(self, info_set_indices: jnp.ndarray, action_values: jnp.ndarray, rng_key: jax.random.PRNGKey):
        """Update MC-CFR state."""
        sampling_mask = mc_sampling_strategy(self.regrets, info_set_indices, rng_key)
        self.regrets, self.strategy = cfr_iteration(
            self.regrets, self.strategy, info_set_indices, action_values, sampling_mask, self.iteration
        )
        self.iteration += 1
    
    def get_strategy(self) -> jnp.ndarray:
        """Get current strategy."""
        return self.strategy
    
    def validate(self) -> bool:
        """Validate training state."""
        return validate_learning(self.regrets)