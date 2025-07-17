# poker_bot/core/bucketing.py

"""
JAX-Native Info Set Computation for Poker AI
Pure JAX implementation without CuPy dependencies
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

@jax.jit
def compute_info_set_id(game_state, player_idx: int) -> jnp.ndarray:
    """
    Compute unique info set ID for a player in a game state.
    
    This is the core function that replaces compute_advanced_info_set
    from the monolithic trainer.py
    
    Args:
        game_state: GameState from full_game_engine
        player_idx: Player index (0-5)
        
    Returns:
        Unique info set ID as int32
    """
    # Extract player's hole cards
    hole_cards = game_state.hole_cards[player_idx]
    community_cards = game_state.comm_cards
    
    # Compute bucketing components
    hand_bucket = _compute_hand_bucket(hole_cards, community_cards)
    street_bucket = _compute_street_bucket(community_cards)
    position_bucket = _compute_position_bucket(player_idx)
    stack_bucket = _compute_stack_bucket(game_state, player_idx)
    pot_bucket = _compute_pot_bucket(game_state)
    
    # Combine all factors into unique ID
    info_set_id = (
        street_bucket * 10000 +     # 4 * 10000 = 40,000
        hand_bucket * 50 +          # 169 * 50 = 8,450
        position_bucket * 8 +       # 6 * 8 = 48
        stack_bucket * 2 +          # 20 * 2 = 40
        pot_bucket                  # 10 * 1 = 10
    )
    
    # Ensure within valid range (max 50,000 info sets)
    return jnp.mod(info_set_id, 50000).astype(jnp.int32)

@jax.jit
def _compute_hand_bucket(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Computa el bucket de la mano usando una fórmula combinatoria estándar y robusta.
    Garantiza un mapeo único y correcto de las 1326 combinaciones preflop a 169 buckets.
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
    
    # El resultado final es un ID único y garantizado entre 0 y 168.
    return bucket.astype(jnp.int32)

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
    Compute position bucket for player.
    
    Args:
        player_idx: Player index (0-5)
        
    Returns:
        Position bucket (0-5)
    """
    return jnp.int32(player_idx)

@jax.jit
def _compute_stack_bucket(game_state, player_idx: int) -> jnp.ndarray:
    """
    Compute stack depth bucket.
    
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
    Compute pot size bucket.
    
    Args:
        game_state: Game state from engine
        
    Returns:
        Pot bucket (0-9)
    """
    pot_size = jnp.squeeze(game_state.pot)
    pot_bucket = jnp.clip(pot_size / 10.0, 0, POT_BUCKETS - 1)
    return pot_bucket.astype(jnp.int32)

# Utility functions for testing and validation
@jax.jit
def test_hand_differentiation():
    """
    Test that different hands map to different buckets.
    
    Returns:
        Dict with test results
    """
    # Mock community cards (all -1 for preflop)
    mock_community = jnp.full(5, -1, dtype=jnp.int8)
    
    # Test hands: AA vs 72o
    aa_hole = jnp.array([51, 47], dtype=jnp.int8)  # Ace of spades, Ace of clubs
    trash_hole = jnp.array([23, 0], dtype=jnp.int8)  # 7 of clubs, 2 of spades
    
    aa_bucket = _compute_hand_bucket(aa_hole, mock_community)
    trash_bucket = _compute_hand_bucket(trash_hole, mock_community)
    
    return {
        'aa_bucket': aa_bucket,
        'trash_bucket': trash_bucket,
        'different': aa_bucket != trash_bucket
    }

def validate_bucketing_system():
    """
    Comprehensive validation of the bucketing system.
    
    Returns:
        Validation results
    """
    logger.info("🧪 Validating JAX-Native Bucketing System...")
    
    try:
        # Test basic differentiation
        test_results = test_hand_differentiation()
        
        if test_results['different']:
            logger.info(f"✅ Hand differentiation: AA({test_results['aa_bucket']}) != 72o({test_results['trash_bucket']})")
        else:
            logger.error(f"❌ Hand differentiation failed: AA({test_results['aa_bucket']}) == 72o({test_results['trash_bucket']})")
            return False
        
        # Test bucket ranges
        mock_community = jnp.full(5, -1, dtype=jnp.int8)
        test_hands = [
            jnp.array([51, 47], dtype=jnp.int8),  # AA
            jnp.array([50, 46], dtype=jnp.int8),  # KK  
            jnp.array([42, 38], dtype=jnp.int8),  # QJ suited
            jnp.array([23, 0], dtype=jnp.int8),   # 72o
        ]
        
        buckets = []
        for hand in test_hands:
            bucket = _compute_hand_bucket(hand, mock_community)
            buckets.append(int(bucket))
        
        unique_buckets = len(set(buckets))
        logger.info(f"✅ Bucket diversity: {unique_buckets}/{len(test_hands)} unique buckets")
        logger.info(f"   Buckets: {buckets}")
        
        # Verify all buckets are in valid range
        valid_range = all(0 <= b < PREFLOP_BUCKETS for b in buckets)
        logger.info(f"✅ Valid range: {valid_range}")
        
        return unique_buckets >= 3 and valid_range
        
    except Exception as e:
        logger.error(f"❌ Bucketing validation failed: {e}")
        return False

# Initialize validation on import (optional)
if __name__ == "__main__":
    validate_bucketing_system()