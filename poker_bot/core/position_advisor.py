import jax
import jax.numpy as jnp

@jax.jit
def apply_position_multipliers(base_strategy: jnp.ndarray, position: int, hole_cards: jnp.ndarray) -> jnp.ndarray:
    """
    FIXED: Apply position multipliers with premium hand override - JAX compatible.
    """
    from .starting_hands import classify_starting_hand
    
    hand_strength = classify_starting_hand(hole_cards)
    
    # PREMIUM HAND OVERRIDE: If hand is premium (>0.8), force aggressive strategy
    is_premium = hand_strength > 0.8
    
    # Premium strategy for AA, KK, etc.
    premium_base = jnp.array([
        0.05,   # FOLD: Very low
        0.05,   # CHECK: Very low
        0.05,   # CALL: Very low
        0.25,   # BET_SMALL
        0.25,   # BET_MED  
        0.25,   # BET_LARGE
        0.05,   # RAISE_SMALL
        0.05,   # RAISE_MED
        0.00    # ALL_IN
    ])
    
    # JAX-compatible position factor for premium hands
    utg_factor = jnp.where(
        position == 0, 0.9,           # UTG: slightly more conservative
        jnp.where(position == 3, 1.1, 1.0)  # BTN: slightly more aggressive, others neutral
    )
    
    premium_strategy = premium_base * utg_factor
    premium_strategy = premium_strategy / jnp.sum(premium_strategy)
    
    # Normal strategy for non-premium hands
    position_factors = jnp.array([
        [1.4, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8],  # UTG
        [1.2, 0.95, 0.95, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9],  # MP
        [1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0],   # CO
        [0.6, 1.1, 1.1, 1.3, 1.3, 1.3, 1.2, 1.2, 1.1],   # BTN
        [1.1, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.95], # SB
        [0.9, 1.05, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # BB
    ])
    
    safe_position = jnp.clip(position, 0, 5)
    adjustments = position_factors[safe_position]
    normal_strategy = base_strategy * adjustments
    normal_sum = jnp.sum(normal_strategy)
    normal_strategy = jnp.where(normal_sum > 0, normal_strategy / normal_sum, base_strategy)
    
    # Return premium strategy for premium hands, normal for others
    return jnp.where(is_premium, premium_strategy, normal_strategy) 