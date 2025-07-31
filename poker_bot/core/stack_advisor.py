import jax
import jax.numpy as jnp

@jax.jit  
def apply_stack_strategy(strategy: jnp.ndarray, stack_size: jnp.ndarray, pot_size: jnp.ndarray) -> jnp.ndarray:
    """
    Adjust strategy based on effective stack depth.
    
    Args:
        strategy: [9] array of action probabilities
        stack_size: Current stack size
        pot_size: Current pot size
        
    Returns:
        stack_adjusted_strategy: [9] array of adjusted probabilities
    """
    stack_scalar = jnp.squeeze(stack_size)
    pot_scalar = jnp.squeeze(pot_size) 
    spr = stack_scalar / jnp.maximum(pot_scalar, 1.0)  # Stack-to-pot ratio
    
    # Stack depth adjustments for each action
    # [FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE, RAISE_SMALL, RAISE_MED, ALL_IN]
    
    def short_stack_adjustments():  # SPR < 3
        return jnp.array([0.8, 0.7, 0.8, 0.5, 0.3, 0.1, 0.3, 0.1, 2.0])  # Favor all-in
        
    def medium_stack_adjustments():  # SPR 3-8  
        return jnp.array([1.0, 1.0, 1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.0])  # Standard aggressive
        
    def deep_stack_adjustments():  # SPR > 8
        return jnp.array([1.1, 1.3, 1.2, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])  # More postflop play
    
    adjustments = jnp.where(
        spr < 3.0, short_stack_adjustments(),
        jnp.where(spr < 8.0, medium_stack_adjustments(), deep_stack_adjustments())
    )
    
    adjusted_strategy = strategy * adjustments
    strategy_sum = jnp.sum(adjusted_strategy)
    return jnp.where(strategy_sum > 0, adjusted_strategy / strategy_sum, strategy) 