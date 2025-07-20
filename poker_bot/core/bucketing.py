# poker_bot/core/bucketing.py

"""
JAX-Native Info Set Computation for Poker AI
Pure JAX implementation without CuPy dependencies
Enhanced for NLHE with improved position and stack awareness
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Constants for bucketing
MAX_HAND_RANK = 12  # Ace high
MAX_SUITS = 4
PREFLOP_BUCKETS = 169  # Standard preflop hand categories
POSITION_BUCKETS = 6   # UTG, UTG+1, MP, CO, BTN, BB
STREET_BUCKETS = 4     # Preflop, Flop, Turn, River
STACK_BUCKETS = 20     # Stack depth categories
POT_BUCKETS = 10       # Pot size categories
ACTION_BUCKETS = 9     # Action history buckets

@jax.jit
def compute_info_set_id(hole_cards: jnp.ndarray, community_cards: jnp.ndarray, 
                       player_idx: int, pot_size: jnp.ndarray = None, 
                       stack_size: jnp.ndarray = None, action_history: jnp.ndarray = None) -> jnp.ndarray:
    """
    Compute unique info set ID for a player given their cards and game state.
    Enhanced for NLHE with improved position and stack awareness.
    
    Args:
        hole_cards: Player's hole cards [2] array
        community_cards: Community cards [5] array (-1 for missing)
        player_idx: Player index (0-5)
        pot_size: Current pot size (optional)
        stack_size: Player's stack size (optional)
        action_history: Recent action history (optional)
        
    Returns:
        Unique info set ID as int32
    """
    
    # Compute bucketing components
    hand_bucket = _compute_hand_bucket(hole_cards, community_cards)
    street_bucket = _compute_street_bucket(community_cards)
    position_bucket = _compute_position_bucket(player_idx)
    
    # Enhanced stack and pot bucketing
    if stack_size is not None:
        stack_bucket = _compute_stack_bucket_enhanced(stack_size, pot_size)
    else:
        stack_bucket = jnp.int32(0)
    
    if pot_size is not None:
        pot_bucket = _compute_pot_bucket_enhanced(pot_size, stack_size)
    else:
        pot_bucket = jnp.int32(0)
    
    # Action history bucketing for NLHE
    if action_history is not None:
        action_bucket = _compute_action_history_bucket(action_history)
    else:
        action_bucket = jnp.int32(0)
    
    # Combine all factors into unique ID with improved weighting
    # Use larger ranges for more important factors
    info_set_id = (
        street_bucket * 100000 +     # 4 * 100000 = 400,000
        hand_bucket * 2000 +         # 169 * 2000 = 338,000
        position_bucket * 300 +      # 6 * 300 = 1,800
        stack_bucket * 15 +          # 20 * 15 = 300
        pot_bucket * 3 +             # 10 * 3 = 30
        action_bucket                # 9 * 1 = 9
    )
    
    # Ensure within valid range (max 500,000 info sets)
    return jnp.mod(info_set_id, 500000).astype(jnp.int32)

@jax.jit
def _compute_hand_bucket(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Computa el bucket de la mano usando una fórmula combinatoria estándar y robusta.
    Garantiza un mapeo único y correcto de las 1326 combinaciones preflop a 169 buckets.
    Enhanced for postflop play.
    """
    # 1. CONVENCIÓN DE RANGO/PALO (COMPATIBLE CON EL CÓDIGO EXISTENTE)
    # Usar la misma convención que el resto del código
    # Rango: card // 4 (0=2, 1=3, ..., 11=K, 12=A)
    # Palo: card % 4 (0=♠, 1=♥, 2=♦, 3=♣)
    # IMPORTANTE: Convertir a int32 para evitar overflow
    ranks = (hole_cards // 4).astype(jnp.int32)
    suits = (hole_cards % 4).astype(jnp.int32)

    # 2. ORDENAR LOS RANGOS
    # Es crucial para que (Rey, Reina) y (Reina, Rey) mapeen al mismo bucket.
    r1 = jnp.maximum(ranks[0], ranks[1]).astype(jnp.int32)  # La carta más alta
    r2 = jnp.minimum(ranks[0], ranks[1]).astype(jnp.int32)  # La carta más baja

    is_pair = (r1 == r2)
    is_suited = (suits[0] == suits[1])

    # 3. FÓRMULA MATEMÁTICA COMBINATORIA (El Corazón de la Solución)
    # Esto no son "números mágicos". Es una forma matemática de asignar un índice único
    # a cada par de cartas.

    def pocket_pair_bucket():
        # Los pares son los más fáciles. Hay 13.
        # AA (rango 12) -> bucket 12
        # KK (rango 11) -> bucket 11
        # ...
        # 22 (rango 0) -> bucket 0
        return r1

    def suited_hand_bucket():
        # Hay 78 manos "suited" únicas (12+11+...+1). Necesitamos un índice único para cada una.
        # La fórmula para la suma de los primeros n-1 números es n*(n-1)/2. Esto nos da un
        # índice base único para cada carta alta (r1). Luego sumamos la carta baja (r2)
        # para diferenciar dentro de ese grupo.
        # Se suma 13 para no solaparse con los buckets de los pares (0-12).
        # Ejemplo KQs (r1=11, r2=10): 13 + (11*10/2) + 10 = 13 + 55 + 10 = 78
        return jnp.int32(13) + (r1 * (r1 - jnp.int32(1)) // jnp.int32(2)) + r2

    def offsuit_hand_bucket():
        # Exactamente la misma lógica que las manos "suited", pero con un
        # desplazamiento adicional de 78 para no solaparse con ellas.
        # Los buckets de manos suited van de 13 a 90 (13+77).
        # Los buckets de manos off-suit irán de 91 a 168.
        return jnp.int32(13) + jnp.int32(78) + (r1 * (r1 - jnp.int32(1)) // jnp.int32(2)) + r2

    # 4. SELECCIÓN DE LA FÓRMULA CORRECTA
    # lax.cond asegura que solo se ejecute una de estas tres lógicas.
    bucket = lax.cond(
        is_pair,
        pocket_pair_bucket,
        lambda: lax.cond(
            is_suited,
            suited_hand_bucket,
            offsuit_hand_bucket
        )
    )
    
    # 5. POSTFLOP ENHANCEMENT: Adjust bucket based on community cards
    num_community = jnp.sum(community_cards >= 0)
    postflop_adjustment = jnp.where(
        num_community >= 3,
        jnp.int32(169),  # Add offset for postflop hands
        jnp.int32(0)
    )
    
    # El resultado final es un ID único y garantizado
    return (bucket + postflop_adjustment).astype(jnp.int32)

@jax.jit
def _compute_street_bucket(community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Determine current street based on community cards.
    
    Args:
        community_cards: [5] array of card indices
        
    Returns:
        Street bucket (0=preflop, 1=flop, 2=turn, 3=river)
    """
    num_community = jnp.sum(community_cards >= 0)
    
    return lax.cond(
        num_community == 0,
        lambda: jnp.int32(0),  # Preflop
        lambda: lax.cond(
            num_community == 3,
            lambda: jnp.int32(1),  # Flop
            lambda: lax.cond(
                num_community == 4,
                lambda: jnp.int32(2),  # Turn
                lambda: jnp.int32(3)   # River
            )
        )
    )

@jax.jit
def _compute_position_bucket(player_idx: int) -> jnp.ndarray:
    """
    Compute position bucket for player with NLHE awareness.
    
    Args:
        player_idx: Player index (0-5)
        
    Returns:
        Position bucket (0-5): UTG, UTG+1, MP, CO, BTN, BB
    """
    return jnp.int32(player_idx)

@jax.jit
def _compute_stack_bucket_enhanced(stack_size: jnp.ndarray, pot_size: jnp.ndarray = None) -> jnp.ndarray:
    """
    Compute stack depth bucket with NLHE awareness.
    
    Args:
        stack_size: Player's stack size
        pot_size: Current pot size (optional)
        
    Returns:
        Stack bucket (0-19)
    """
    stack_value = jnp.squeeze(stack_size)
    
    if pot_size is not None:
        pot_value = jnp.squeeze(pot_size)
        # Stack-to-pot ratio bucketing
        spr = stack_value / jnp.maximum(pot_value, 1.0)
        
        # NLHE-specific stack depth categories
        bucket = jnp.where(
            spr < 1.0, jnp.int32(0),  # All-in or near all-in
            jnp.where(
                spr < 2.0, jnp.int32(1),  # Short stack
                jnp.where(
                    spr < 4.0, jnp.int32(2),  # Medium stack
                    jnp.where(
                        spr < 8.0, jnp.int32(3),  # Deep stack
                        jnp.int32(4)  # Very deep stack
                    )
                )
            )
        )
        
        # Add position-based adjustment
        position_factor = jnp.int32(5)  # Base multiplier
        return bucket * position_factor
    else:
        # Fallback to absolute stack size
        return jnp.clip(stack_value / 50.0, 0, STACK_BUCKETS - 1).astype(jnp.int32)

@jax.jit
def _compute_pot_bucket_enhanced(pot_size: jnp.ndarray, stack_size: jnp.ndarray = None) -> jnp.ndarray:
    """
    Compute pot size bucket with NLHE awareness.
    
    Args:
        pot_size: Current pot size
        stack_size: Player's stack size (optional)
        
    Returns:
        Pot bucket (0-9)
    """
    pot_value = jnp.squeeze(pot_size)
    
    if stack_size is not None:
        stack_value = jnp.squeeze(stack_size)
        # Pot-to-stack ratio bucketing
        psr = pot_value / jnp.maximum(stack_value, 1.0)
        
        # NLHE-specific pot size categories
        bucket = jnp.where(
            psr < 0.25, jnp.int32(0),  # Small pot
            jnp.where(
                psr < 0.5, jnp.int32(1),  # Medium pot
                jnp.where(
                    psr < 1.0, jnp.int32(2),  # Large pot
                    jnp.where(
                        psr < 2.0, jnp.int32(3),  # Very large pot
                        jnp.int32(4)  # Massive pot
                    )
                )
            )
        )
        
        return bucket
    else:
        # Fallback to absolute pot size
        return jnp.clip(pot_value / 10.0, 0, POT_BUCKETS - 1).astype(jnp.int32)

@jax.jit
def _compute_action_history_bucket(action_history: jnp.ndarray) -> jnp.ndarray:
    """
    Compute action history bucket for NLHE.
    
    Args:
        action_history: Recent action history
        
    Returns:
        Action bucket (0-8)
    """
    # Simple hash of recent actions
    if action_history is None or action_history.size == 0:
        return jnp.int32(0)
    
    # Take last few actions and create a simple hash
    recent_actions = action_history[-3:]  # Last 3 actions
    valid_actions = jnp.where(recent_actions >= 0, recent_actions, 0)
    
    # Create hash from action sequence
    action_hash = jnp.sum(valid_actions * jnp.array([1, 3, 9]))
    
    return jnp.mod(action_hash, ACTION_BUCKETS).astype(jnp.int32)

@jax.jit
def _compute_stack_bucket(game_state, player_idx: int) -> jnp.ndarray:
    """
    Compute stack depth bucket (legacy function for backward compatibility).
    
    Args:
        game_state: Game state from engine
        player_idx: Player index
        
    Returns:
        Stack bucket (0-19)
    """
    # Use pot size as proxy for stack depth
    pot_size = jnp.squeeze(game_state.pot)
    stack_bucket = jnp.clip(pot_size / 5.0, 0, STACK_BUCKETS - 1)
    return stack_bucket.astype(jnp.int32)

@jax.jit
def _compute_pot_bucket(game_state) -> jnp.ndarray:
    """
    Compute pot size bucket (legacy function for backward compatibility).
    
    Args:
        game_state: Game state from engine
        
    Returns:
        Pot bucket (0-9)
    """
    pot_size = jnp.squeeze(game_state.pot)
    pot_bucket = jnp.clip(pot_size / 10.0, 0, POT_BUCKETS - 1)
    return pot_bucket.astype(jnp.int32)

# ==============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ==============================================================================

@jax.jit
def test_hand_differentiation():
    """
    Test that different hands get different buckets.
    """
    # Test pairs
    aa = jnp.array([48, 49])  # AA
    kk = jnp.array([44, 45])  # KK
    
    # Test suited hands
    aks = jnp.array([48, 44])  # AK suited
    aqs = jnp.array([48, 40])  # AQ suited
    
    # Test offsuit hands
    ako = jnp.array([48, 43])  # AK offsuit
    aqo = jnp.array([48, 39])  # AQ offsuit
    
    buckets = jax.vmap(lambda cards: _compute_hand_bucket(cards, jnp.full(5, -1)))(
        jnp.stack([aa, kk, aks, aqs, ako, aqo])
    )
    
    # Check that all buckets are different by comparing each pair
    # This is JAX-compatible and doesn't use jnp.unique
    def check_unique(buckets):
        # Compare each bucket with all others
        def compare_with_others(idx):
            current = buckets[idx]
            others = jnp.concatenate([buckets[:idx], buckets[idx+1:]])
            return jnp.all(current != others)
        
        # Check uniqueness for each bucket
        uniqueness_checks = jax.vmap(compare_with_others)(jnp.arange(len(buckets)))
        return jnp.all(uniqueness_checks)
    
    return check_unique(buckets)

def validate_bucketing_system():
    """
    Validate the entire bucketing system.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Test hand differentiation
        if not test_hand_differentiation():
            logger.error("❌ Hand differentiation test failed")
            return False
        
        # Test info set ID generation
        hole_cards = jnp.array([48, 49])  # AA
        community_cards = jnp.full(5, -1)  # Preflop
        player_idx = 0
        pot_size = jnp.array([15.0])
        stack_size = jnp.array([1000.0])
        
        info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
        
        if info_set_id < 0 or info_set_id >= 500000:
            logger.error(f"❌ Invalid info set ID: {info_set_id}")
            return False
        
        logger.info("✅ Bucketing system validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bucketing system validation failed: {e}")
        return False

# ==============================================================================
# UTILITY FUNCTIONS FOR NLHE
# ==============================================================================

@jax.jit
def compute_nlhe_info_set_features(hole_cards: jnp.ndarray, community_cards: jnp.ndarray,
                                  player_idx: int, pot_size: jnp.ndarray, 
                                  stack_size: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute comprehensive NLHE info set features.
    
    Returns:
        Dictionary of features for NLHE analysis
    """
    hand_bucket = _compute_hand_bucket(hole_cards, community_cards)
    street_bucket = _compute_street_bucket(community_cards)
    position_bucket = _compute_position_bucket(player_idx)
    stack_bucket = _compute_stack_bucket_enhanced(stack_size, pot_size)
    pot_bucket = _compute_pot_bucket_enhanced(pot_size, stack_size)
    
    # Compute additional NLHE-specific features
    stack_to_pot_ratio = jnp.squeeze(stack_size) / jnp.maximum(jnp.squeeze(pot_size), 1.0)
    hand_strength = _compute_hand_strength_estimate(hole_cards, community_cards)
    
    return {
        'hand_bucket': hand_bucket,
        'street_bucket': street_bucket,
        'position_bucket': position_bucket,
        'stack_bucket': stack_bucket,
        'pot_bucket': pot_bucket,
        'stack_to_pot_ratio': stack_to_pot_ratio,
        'hand_strength': hand_strength
    }

@jax.jit
def _compute_hand_strength_estimate(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a simple hand strength estimate for NLHE.
    
    Returns:
        Hand strength estimate (0.0 to 1.0)
    """
    ranks = (hole_cards // 4).astype(jnp.int32)
    suits = (hole_cards % 4).astype(jnp.int32)
    
    # High card value
    high_card_value = jnp.max(ranks) / 12.0
    
    # Pair bonus
    pair_bonus = jnp.where(ranks[0] == ranks[1], 0.3, 0.0)
    
    # Suited bonus
    suited_bonus = jnp.where(suits[0] == suits[1], 0.1, 0.0)
    
    # Connected bonus
    rank_diff = jnp.abs(ranks[0] - ranks[1])
    connected_bonus = jnp.where(rank_diff <= 2, 0.1, 0.0)
    
    # Combine all factors
    strength = high_card_value + pair_bonus + suited_bonus + connected_bonus
    
    return jnp.clip(strength, 0.0, 1.0)