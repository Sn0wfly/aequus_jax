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
    sampling_rate = 1.0  # Process 50% of learning opportunities (was 0.15)
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
    sampling_mask: jnp.ndarray,
    learning_rate: float = 0.0001
) -> jnp.ndarray:
    """FIXED: Proper regret accumulation with dtype consistency and learning rate."""
    # CRITICAL FIX: Ensure consistent dtypes to prevent broadcasting bugs
    info_set_indices = info_set_indices.astype(jnp.int32)
    regrets = regrets.astype(jnp.float32)
    action_regrets = action_regrets.astype(jnp.float32)
    sampling_mask = sampling_mask.astype(jnp.bool_)
    # Only process sampled info sets
    valid_mask = sampling_mask & (info_set_indices >= 0) & (info_set_indices < regrets.shape[0])
    # Get valid indices and regrets, apply learning rate
    valid_indices = jnp.where(valid_mask, info_set_indices, 0).astype(jnp.int32)
    valid_regrets = jnp.where(valid_mask[:, None], action_regrets * learning_rate, jnp.zeros_like(action_regrets)).astype(jnp.float32)
    # Ensure scatter_add parameters are correct
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    # Use scatter_add with proper type casting
    updated_regrets = jax.lax.scatter_add(
        regrets,
        valid_indices[:, None],  # Already cast to int32 above
        valid_regrets,           # Already cast to float32 above
        dimension_numbers=dimension_numbers,
        indices_are_sorted=False,
        unique_indices=False
    )
    updated_regrets = jnp.clip(updated_regrets, -1000.0, 1000.0)
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
    iteration: int,
    learning_rate: float,
    use_discounting: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single CFR iteration with MC sampling."""
    
    # 1. Update regrets (con el learning_rate correcto)
    new_regrets = accumulate_regrets_fixed(regrets, info_set_indices, action_values, sampling_mask, learning_rate)
    
    # 2. APLICAR DESCUENTO (LA PARTE CLAVE QUE FALTABA)
    def apply_discount(r):
        return apply_cfr_plus_discounting(r, iteration)
    
    def no_discount(r):
        return r

    # Usar lax.cond para que sea compatible con JIT
    new_regrets = lax.cond(use_discounting, apply_discount, no_discount, new_regrets)
    
    # 3. Update strategy (sin cambios)
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

@jax.jit
def calculate_strategy_optimized(regrets: jnp.ndarray, visited_mask: jnp.ndarray) -> jnp.ndarray:
    """OPTIMIZED: Only calculate strategy for visited info sets."""
    # jax.debug.print("🔧 calculate_strategy_optimized starting...")
    # jax.debug.print("  regrets shape: {}, visited_mask sum: {}", regrets.shape, jnp.sum(visited_mask))
    # CLAVE: Solo procesar info sets que han sido visitados
    positive_regrets = jnp.maximum(regrets, 0.0)
    sum_positive = jnp.sum(positive_regrets, axis=-1, keepdims=True)
    # Strategy uniforme por defecto
    uniform_strategy = jnp.ones_like(regrets) / regrets.shape[-1]
    # OPTIMIZACIÓN: Solo calcular para info sets visitados
    strategy = jnp.where(
        visited_mask[:, None] & (sum_positive > 1e-12),
        positive_regrets / (sum_positive + 1e-12),
        uniform_strategy
    )
    # jax.debug.print("✅ calculate_strategy_optimized completed")
    return strategy

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