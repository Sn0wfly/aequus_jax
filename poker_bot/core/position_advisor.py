import jax
import jax.numpy as jnp

@jax.jit
def apply_position_multipliers(base_strategy: jnp.ndarray, position: int, hole_cards: jnp.ndarray) -> jnp.ndarray:
    """
    FIXED: Apply position multipliers with premium hand override - JAX compatible.
    """
    from .starting_hands import classify_starting_hand
    
    hand_strength = classify_starting_hand(hole_cards)
    
    # Detect pocket pairs for special handling
    ranks = hole_cards // 4
    is_pocket_pair = ranks[0] == ranks[1]
    
    # PREMIUM HAND OVERRIDE: Position-aware premium detection
    # For UTG, exclude JJ specifically (JJ has rank 9)
    is_premium_utg = (hand_strength >= 0.95) & (ranks[0] >= 10)  # AA, KK, QQ only (exclude JJ)
    # For BTN+, allow AK, AQ, KQ (all have 0.85 strength)
    is_premium_other = jnp.where(
        position <= 2,  # MP, CO  
        hand_strength >= 0.75,  # FIXED: >= instead of >
        hand_strength >= 0.75   # FIXED: Lower threshold for BTN+
    )
    is_premium = jnp.where(position == 0, is_premium_utg, is_premium_other)
    
    # JJ (0.95) should be considered premium from BTN+ positions only
    # Only JJ and better pocket pairs, not TT/99
    is_pocket_pair_premium = (hand_strength >= 0.95) & is_pocket_pair & (position >= 3)
    is_premium = is_premium | is_pocket_pair_premium
    
    # Premium strategy for AA, KK, etc. - ENHANCED
    premium_base = jnp.array([
        0.05,   # FOLD: Very low
        0.05,   # CHECK: Very low
        0.10,   # CALL: Low
        0.20,   # BET_SMALL
        0.25,   # BET_MED  
        0.25,   # BET_LARGE
        0.05,   # RAISE_SMALL
        0.05,   # RAISE_MED
        0.00    # ALL_IN
    ])
    
    # JAX-compatible position factor for premium hands - POSITION AWARE
    utg_factor = jnp.where(
        position == 0, 0.8,           # UTG: conservative but still aggressive
        jnp.where(position == 3, 1.2, 1.0)  # BTN: more aggressive, others neutral
    )
    
    premium_strategy = premium_base * utg_factor
    premium_strategy = premium_strategy / jnp.sum(premium_strategy)
    
    # Professional position factors - BALANCED RANGES
    position_factors = jnp.array([
        # [FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE, RAISE_SMALL, RAISE_MED, ALL_IN]
        [1.5, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8],   # UTG - BALANCED
        [1.3, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9],   # MP - MODERATE
        [1.1, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0],   # CO - STANDARD
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # BTN - NEUTRAL
        [1.4, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8],   # SB - DEFENSIVE
        [1.2, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]    # BB - DEFENSIVE
    ])
    
    safe_position = jnp.clip(position, 0, 5)
    adjustments = position_factors[safe_position]
    normal_strategy = base_strategy * adjustments
    normal_sum = jnp.sum(normal_strategy)
    normal_strategy = jnp.where(normal_sum > 0, normal_strategy / normal_sum, base_strategy)
    
    # Force fold trash hands regardless of position
    is_trash = hand_strength < 0.30
    trash_strategy = jnp.array([0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    
    # Return appropriate strategy based on hand quality
    return jnp.where(
        is_trash, trash_strategy,
        jnp.where(is_premium, premium_strategy, normal_strategy)
    ) 