import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def classify_starting_hand(hole_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Clasifica las hole cards según estándares profesionales.
    
    Returns:
        hand_strength: float entre 0.0-1.0
    """
    ranks = hole_cards // 4  # 0-12 (2=0, 3=1, ..., A=12)
    suits = hole_cards % 4
    
    # Ordenar ranks de mayor a menor
    high_rank = jnp.max(ranks)
    low_rank = jnp.min(ranks)
    is_suited = suits[0] == suits[1]
    is_pair = ranks[0] == ranks[1]
    
    # PREMIUM HANDS (0.85-1.0)
    premium_pairs = (is_pair & (high_rank >= 9))  # AA, KK, QQ, JJ
    premium_unpaired = (
        ((high_rank == 12) & (low_rank >= 10)) |  # AK, AQ, AJ  
        ((high_rank == 11) & (low_rank >= 10))    # KQ, KJ
    )
    
    # STRONG HANDS (0.65-0.84)
    strong_pairs = (is_pair & (high_rank >= 6) & (high_rank <= 8))  # 77-99
    strong_aces = ((high_rank == 12) & (low_rank >= 7) & (low_rank <= 9))  # AT-A9
    strong_suited = (is_suited & ((high_rank >= 9) & (low_rank >= 6)))  # Suited broadway
    
    # MEDIUM HANDS (0.35-0.64)
    medium_pairs = (is_pair & (high_rank >= 2) & (high_rank <= 5))  # 22-66
    medium_aces = ((high_rank == 12) & (low_rank >= 2) & (low_rank <= 6))  # A2-A8
    suited_connectors = (is_suited & (high_rank - low_rank <= 4) & (low_rank >= 3))  # 65s+
    
    # WEAK HANDS (0.1-0.34)
    # Todo lo demás
    
    return jnp.where(
        premium_pairs, 0.95,  # AA, KK, QQ, JJ
        jnp.where(
            premium_unpaired & is_suited, 0.85,  # AKs, AQs, AJs, KQs
            jnp.where(
                premium_unpaired, 0.80,  # AKo, AQo, AJo, KQo
                jnp.where(
                    strong_pairs, 0.70,  # TT, 99, 88, 77
                    jnp.where(
                        strong_aces & is_suited, 0.65,  # ATs, A9s, A8s
                        jnp.where(
                            strong_suited, 0.60,  # Suited broadway
                            jnp.where(
                                medium_pairs, 0.45,  # 66, 55, 44, 33, 22
                                jnp.where(
                                    suited_connectors, 0.50,  # 65s, 54s, etc.
                                    jnp.where(
                                        medium_aces, 0.40,  # A7o, A6o, etc.
                                        0.20  # Trash hands
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

@jax.jit
def get_position_multiplier(position: int) -> float:
    """
    Multiplica hand strength basado en posición.
    0=UTG (tight), 1=MP, 2=CO, 3=BTN (loose), 4=SB, 5=BB
    """
    # Valores profesionales: UTG más tight, BTN más loose
    position_values = jnp.array([
        0.75,  # UTG - Very tight (fold 85% of hands)
        0.85,  # MP - Tight 
        0.95,  # CO - Standard
        1.20,  # BTN - Loose (play 60% more hands)
        0.90,  # SB - Slightly tight (bad position post-flop)
        0.95   # BB - Standard (already invested)
    ])
    
    # Clamp position to valid range
    safe_position = jnp.clip(position, 0, 5)
    return position_values[safe_position]

@jax.jit
def classify_starting_hand_with_position(hole_cards: jnp.ndarray, position: int) -> jnp.ndarray:
    """
    Clasifica starting hand con position awareness.
    
    Args:
        hole_cards: Array de 2 cartas
        position: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
    
    Returns:
        hand_strength ajustada por posición (0.0-1.0)
    """
    base_strength = classify_starting_hand(hole_cards)
    position_multiplier = get_position_multiplier(position)
    
    # Aplicar position multiplier
    adjusted_strength = base_strength * position_multiplier
    
    # Clamp para mantener rango válido
    return jnp.clip(adjusted_strength, 0.05, 0.98)

@jax.jit
def evaluate_hand_strength_multi_street(
    hole_cards: jnp.ndarray, 
    community_cards: jnp.ndarray, 
    position: int
) -> jnp.ndarray:
    """
    Evaluación completa multi-street con board texture.
    """
    from .board_analysis import analyze_board_texture, get_street_multiplier, analyze_hand_vs_board
    
    num_community = jnp.sum(community_cards >= 0)
    
    # PREFLOP: Usar starting hands con position
    preflop_strength = classify_starting_hand_with_position(hole_cards, position)
    
    # POST-FLOP: Combinar hand vs board + board texture
    postflop_base = analyze_hand_vs_board(hole_cards, community_cards)
    board_wetness = analyze_board_texture(community_cards)
    street_multiplier = get_street_multiplier(num_community)
    
    # En boards wet, ser más cauteloso con manos marginales
    board_adjustment = jnp.where(
        board_wetness > 0.6,  # Board muy wet
        postflop_base * 0.9,  # Reducir confianza
        postflop_base * 1.1   # Board dry, aumentar confianza
    )
    
    postflop_strength = board_adjustment * street_multiplier
    
    # Seleccionar evaluación según street
    final_strength = jnp.where(
        num_community == 0,  # Preflop
        preflop_strength,
        postflop_strength
    )
    
    return jnp.clip(final_strength, 0.05, 0.98) 