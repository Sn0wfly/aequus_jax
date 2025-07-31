import jax
import jax.numpy as jnp

@jax.jit
def apply_position_multipliers(base_strategy: jnp.ndarray, position: int, hole_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Adjust strategy probabilities based on position.
    
    Args:
        base_strategy: [9] array of action probabilities from bot.strategy[info_set_id]
        position: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB  
        hole_cards: [2] array of hole cards
        
    Returns:
        position_adjusted_strategy: [9] array of adjusted probabilities
    """
    from .starting_hands import classify_starting_hand
    
    hand_strength = classify_starting_hand(hole_cards)
    
    # Position factors: UTG tight, BTN loose
    position_factors = jnp.array([
        [1.4, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8],  # UTG: more fold, less aggressive
        [1.2, 0.95, 0.95, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9],  # MP
        [1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0],   # CO: baseline
        [0.6, 1.1, 1.1, 1.3, 1.3, 1.3, 1.2, 1.2, 1.1],   # BTN: less fold, more aggressive  
        [1.1, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.95], # SB
        [0.9, 1.05, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # BB
    ])
    
    # Apply stronger position effects for marginal hands
    strength_multiplier = jnp.where(
        hand_strength < 0.4,  # Marginal hands
        1.5,  # Stronger position effect
        jnp.where(hand_strength > 0.8, 0.7, 1.0)  # Premium hands less affected
    )
    
    safe_position = jnp.clip(position, 0, 5)
    adjustments = position_factors[safe_position] 
    final_adjustments = 1.0 + (adjustments - 1.0) * strength_multiplier
    
    adjusted_strategy = base_strategy * final_adjustments
    
    # Renormalize to valid probabilities
    strategy_sum = jnp.sum(adjusted_strategy)
    return jnp.where(strategy_sum > 0, adjusted_strategy / strategy_sum, base_strategy) 