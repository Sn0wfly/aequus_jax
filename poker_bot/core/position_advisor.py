import jax
import jax.numpy as jnp

@jax.jit
def apply_position_multipliers(base_strategy: jnp.ndarray, position: int, hole_cards: jnp.ndarray) -> jnp.ndarray:
    """
    FIXED: Apply position multipliers with premium hand override.
    """
    from .starting_hands import classify_starting_hand
    
    hand_strength = classify_starting_hand(hole_cards)
    
    # PREMIUM HAND OVERRIDE: If hand is premium (>0.8), force aggressive strategy
    is_premium = hand_strength > 0.8
    
    def premium_strategy():
        # Override with aggressive strategy for premium hands
        premium_strat = jnp.array([
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
        
        # Slight position adjustment for premium hands
        utg_factor = 0.9 if position == 0 else 1.1 if position == 3 else 1.0
        return premium_strat * utg_factor / jnp.sum(premium_strat * utg_factor)
    
    def normal_strategy():
        # Original logic for non-premium hands
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
        adjusted_strategy = base_strategy * adjustments
        strategy_sum = jnp.sum(adjusted_strategy)
        return jnp.where(strategy_sum > 0, adjusted_strategy / strategy_sum, base_strategy)
    
    # Use premium strategy for premium hands, normal for others
    return jnp.where(is_premium, premium_strategy(), normal_strategy()) 