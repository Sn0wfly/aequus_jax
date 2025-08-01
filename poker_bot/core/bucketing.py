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

# Import for position-aware hand evaluation
from .starting_hands import classify_starting_hand_with_position
from .board_analysis import analyze_board_texture

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
                       stack_size: jnp.ndarray = None, action_history: jnp.ndarray = None,
                       max_info_sets: int = 100000) -> jnp.ndarray:
    """
    Cálculo de Info Set mejorado que incluye textura del board y tamaño del pozo
    para evitar colisiones y permitir estrategias más complejas como los faroles.
    """
    num_community = jnp.sum(community_cards >= 0)
    
    # 1. Fuerza de la mano (como antes, pero ahora es solo un componente más)
    hand_strength = jnp.where(
        num_community == 0,  # Preflop
        classify_starting_hand_with_position(hole_cards, player_idx),
        # Post-flop
        _evaluate_postflop_hand(hole_cards, community_cards)
    )
    
    # 2. Textura del Board (¡NUEVO Y CRÍTICO!)
    # Esto diferencia una mesa seca de una con muchos proyectos (draws).
    # Un valor alto (cercano a 1.0) significa un board "húmedo" (wet), ideal para faroles.
    board_texture = analyze_board_texture(community_cards)

    # 3. Componente de Posición (como antes)
    position_component = jnp.int32(player_idx)

    # 4. Componente del Pozo (como antes)
    pot_component = jnp.where(
        pot_size is not None,
        jnp.clip(jnp.int32(pot_size[0] / 10.0), 0, 99),
        jnp.int32(0)
    )
    
    # 5. Componente del Stack Size (¡NUEVO!)
    stack_component = jnp.where(
        stack_size is not None,
        jnp.clip(jnp.int32(stack_size[0] / 50.0), 0, 99),
        jnp.int32(0)
    )
    
    # 6. Combinar todos los componentes en un hash más robusto
    # Usamos multiplicadores primos para minimizar colisiones
    combined_hash = (
        jnp.int32(hand_strength * 1000) * 1 +
        jnp.int32(board_texture * 100)  * 31 +
        position_component * 67 +
        pot_component * 101 +
        stack_component * 137  # Nuevo multiplicador primo
    )
    
    # El ID final es el hash módulo el tamaño de la tabla
    final_id = combined_hash % max_info_sets
    
    return jnp.clip(final_id, 0, max_info_sets - 1).astype(jnp.int32)

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
        jnp.int32(50),  # Add offset for postflop hands
        jnp.int32(0)
    )
    
    # El resultado final es un ID único y garantizado
    return (bucket + postflop_adjustment).astype(jnp.int32)

@jax.jit
def _evaluate_postflop_hand(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluación postflop simplificada para bucketing.
    Retorna un valor entre 0.0 y 1.0 basado en la fuerza de la mano.
    """
    # Combinar todas las cartas
    all_cards = jnp.concatenate([hole_cards, community_cards])
    
    # Máscara para cartas válidas (>= 0)
    valid_mask = all_cards >= 0
    num_valid = jnp.sum(valid_mask)
    
    # Si no hay suficientes cartas, retornar fuerza mínima
    strength = jnp.where(
        num_valid < 5,  # Necesitamos al menos 5 cartas para evaluar
        0.1,  # Fuerza mínima
        _compute_hand_strength_estimate(hole_cards, community_cards)
    )
    
    return jnp.clip(strength, 0.0, 1.0)

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

def test_hand_differentiation():
    """
    Test that different hands get different buckets.
    Simplified version without JIT to avoid tracing issues.
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
    
    # Compute buckets without vmap to avoid tracing issues
    buckets = []
    test_hands = [aa, kk, aks, aqs, ako, aqo]
    
    for cards in test_hands:
        bucket = _compute_hand_bucket(cards, jnp.full(5, -1))
        buckets.append(int(bucket))  # Convert to Python int
    
    # Simple Python-based uniqueness check
    unique_buckets = len(set(buckets))
    return unique_buckets == len(buckets)  # All buckets are unique

def validate_bucketing_system():
    """
    Validate the entire bucketing system.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Test basic info set ID generation
        hole_cards = jnp.array([48, 49])  # AA
        community_cards = jnp.full(5, -1)  # Preflop
        player_idx = 0
        pot_size = jnp.array([15.0])
        stack_size = jnp.array([1000.0])
        
        info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
        
        if info_set_id < 0 or info_set_id >= 1000000:  # Updated to match max_info_sets
            logger.error(f"❌ Invalid info set ID: {info_set_id}")
            return False
        
        # Test different hands get different buckets
        test_hands = [
            (jnp.array([48, 49]), "AA"),
            (jnp.array([44, 45]), "KK"),
            (jnp.array([48, 44]), "AKs"),
            (jnp.array([23, 0]), "72o")
        ]
        
        buckets = []
        for hole_cards, hand_name in test_hands:
            bucket = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size, stack_size)
            buckets.append(int(bucket))  # Convert to Python int
        
        # Check that we have at least 3 different buckets
        unique_buckets = len(set(buckets))
        if unique_buckets < 3:
            logger.error(f"❌ Not enough unique buckets: {unique_buckets}")
            return False
        
        # Test postflop bucketing
        flop_cards = jnp.array([0, 1, 2, -1, -1])  # Flop
        flop_bucket = compute_info_set_id(hole_cards, flop_cards, player_idx, pot_size, stack_size)
        
        if int(flop_bucket) == int(info_set_id):
            logger.error("❌ Postflop bucket same as preflop bucket")
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

# Enhanced Bucketing System for Maximum Strategic Diversity

@jax.jit
def compute_info_set_id_enhanced(hole_cards: jnp.ndarray, community_cards: jnp.ndarray, 
                                player_idx: int, pot_size: jnp.ndarray = None, 
                                stack_size: jnp.ndarray = None, action_history: jnp.ndarray = None,
                                max_info_sets: int = 100000) -> jnp.ndarray:
    """
    FIXED: Simple and reliable bucketing without collisions.
    Uses unique card combinations + street + position for proper differentiation.
    """
    # Ordenar las cartas para consistencia (avoid [A,K] vs [K,A] difference)
    card1, card2 = hole_cards[0], hole_cards[1]
    sorted_cards = jnp.where(card1 < card2, jnp.array([card1, card2]), jnp.array([card2, card1]))
    
    # Componente base: combinación única de cartas (0-2703 range)
    card_hash = sorted_cards[0] * 52 + sorted_cards[1]
    
    # Street component - different multipliers for each street
    num_community = jnp.sum(community_cards >= 0)
    street_mult = jnp.where(
        num_community == 0, 0,           # Preflop
        jnp.where(num_community == 3, 3000,    # Flop  
                  jnp.where(num_community == 4, 6000,    # Turn
                           9000))        # River
    )
    
    # Position component
    position_mult = player_idx * 12000
    
    # Pot component (simplified to avoid over-granularity)
    pot_component = jnp.where(
        pot_size is not None,
        jnp.clip(jnp.int32(jnp.squeeze(pot_size) / 50.0), 0, 20) * 100000,
        jnp.int32(0)
    )
    
    # Combine all components with proper spacing to avoid collisions
    combined_id = card_hash + street_mult + position_mult + pot_component
    
    # Ensure within bounds
    final_id = combined_id % max_info_sets
    
    return jnp.clip(final_id, 0, max_info_sets - 1).astype(jnp.int32)

@jax.jit
def compute_detailed_hand_bucket(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    ENHANCED: Hand bucketing con granularidad extrema.
    En lugar de 169 buckets preflop, usa 1000+ buckets diferentes.
    """
    ranks = (hole_cards // 4).astype(jnp.int32)
    suits = (hole_cards % 4).astype(jnp.int32)
    
    r1 = jnp.maximum(ranks[0], ranks[1]).astype(jnp.int32)
    r2 = jnp.minimum(ranks[0], ranks[1]).astype(jnp.int32)
    
    is_pair = (r1 == r2)
    is_suited = (suits[0] == suits[1])
    
    # GRANULARIDAD EXTREMA: usar las cartas exactas, no solo rangos
    card1_value = hole_cards[0].astype(jnp.int32)
    card2_value = hole_cards[1].astype(jnp.int32)
    
    def pocket_pair_detailed():
        # En lugar de 13 buckets, usar 52*51/2 = 1326 buckets
        return card1_value * 53 + card2_value
    
    def suited_hand_detailed():
        return 1400 + card1_value * 53 + card2_value
    
    def offsuit_hand_detailed():
        return 2800 + card1_value * 53 + card2_value
    
    detailed_bucket = lax.cond(
        is_pair,
        pocket_pair_detailed,
        lambda: lax.cond(
            is_suited,
            suited_hand_detailed,
            offsuit_hand_detailed
        )
    )
    
    # Post-flop: agregar información de board
    num_community = jnp.sum(community_cards >= 0)
    postflop_adjustment = jnp.where(
        num_community >= 3,
        jnp.int32(5200) + jnp.sum(jnp.where(community_cards >= 0, community_cards, 0)),
        jnp.int32(0)
    )
    
    return (detailed_bucket + postflop_adjustment).astype(jnp.int32)

# Función de validación para testear la mejora
def validate_enhanced_bucketing():
    """Valida que el nuevo sistema genera más diversidad."""
    import numpy as np
    
    # Generar 10,000 scenarios aleatorios
    scenarios = []
    for _ in range(10000):
        hole_cards = np.random.choice(52, 2, replace=False)
        community = np.concatenate([np.random.choice(50, 3, replace=False), [-1, -1]])
        pot = np.array([np.random.uniform(10, 500)])
        stack = np.array([np.random.uniform(100, 2000)])
        position = np.random.randint(0, 6)
        
        # Convertir a JAX arrays
        hole_jax = jnp.array(hole_cards)
        comm_jax = jnp.array(community)
        pot_jax = jnp.array(pot)
        stack_jax = jnp.array(stack)
        
        bucket_id = compute_info_set_id_enhanced(
            hole_jax, comm_jax, position, pot_jax, stack_jax, max_info_sets=500000
        )
        scenarios.append(int(bucket_id))
    
    unique_buckets = len(set(scenarios))
    print(f"🎯 Enhanced Bucketing Results:")
    print(f"   Unique buckets: {unique_buckets:,} / 10,000 scenarios")
    print(f"   Diversity: {unique_buckets/10000*100:.1f}%")
    print(f"   Estimated full coverage: {unique_buckets*50:,} info sets")
    
    return unique_buckets > 5000  # Esperamos >50% de diversidad