import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def analyze_board_texture(community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Analiza textura del board: wet (draws) vs dry (estático).
    100% compatible con JAX JIT - sin if statements.
    """
    # Contar cartas válidas
    num_cards = jnp.sum(community_cards >= 0)
    
    # Procesar cartas (máximo 5)
    max_cards = jnp.minimum(num_cards, 5)
    
    # FLUSH DRAWS
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(5):
        valid_card = jnp.where(i < max_cards, community_cards[i], -1)
        suit = jnp.where(valid_card >= 0, valid_card % 4, 0)
        suit_counts = jnp.where(
            valid_card >= 0,
            suit_counts.at[suit].add(1),
            suit_counts
        )
    
    max_suit_count = jnp.max(suit_counts)
    flush_draw_strength = jnp.where(
        max_suit_count >= 3, 0.4,
        jnp.where(max_suit_count == 2, 0.2, 0.0)
    )
    
    # STRAIGHT DRAWS
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(5):
        valid_card = jnp.where(i < max_cards, community_cards[i], -1)
        rank = jnp.where(valid_card >= 0, valid_card // 4, 0)
        rank_counts = jnp.where(
            valid_card >= 0,
            rank_counts.at[rank].add(1),
            rank_counts
        )
    
    # Detectar secuencias
    consecutive_ranks = 0
    for start_rank in range(11):
        has_rank = rank_counts[start_rank] > 0
        has_next = rank_counts[start_rank + 1] > 0
        has_after = rank_counts[start_rank + 2] > 0
        consecutive_ranks = jnp.where(
            has_rank & has_next,
            jnp.maximum(consecutive_ranks, 2),
            consecutive_ranks
        )
        consecutive_ranks = jnp.where(
            has_rank & has_next & has_after,
            jnp.maximum(consecutive_ranks, 3),
            consecutive_ranks
        )
    
    straight_potential = jnp.where(consecutive_ranks >= 2, 0.3, 0.0)
    
    # PAIRED BOARDS
    max_rank_count = jnp.max(rank_counts)
    pairs_count = jnp.sum(rank_counts >= 2)
    
    pair_texture = jnp.where(
        max_rank_count >= 3, 0.8,
        jnp.where(pairs_count >= 2, 0.6,
                  jnp.where(max_rank_count == 2, 0.3, 0.0))
    )
    
    # HIGH CARDS
    high_cards = jnp.sum(rank_counts[9:13])
    high_card_factor = jnp.where(
        num_cards > 0,
        (high_cards / num_cards) * 0.2,
        0.0
    )
    
    # COMBINAR FACTORES
    total_wetness = (
        flush_draw_strength + 
        straight_potential + 
        pair_texture + 
        high_card_factor
    )
    
    # Si menos de 3 cartas, retornar neutral (0.5)
    final_wetness = jnp.where(
        num_cards >= 3,
        jnp.clip(total_wetness, 0.0, 1.0),
        0.5
    )
    
    return final_wetness

@jax.jit
def get_street_multiplier(num_community_cards: int) -> float:
    """
    Multiplier basado en la calle actual.
    
    Args:
        num_community_cards: 0=preflop, 3=flop, 4=turn, 5=river
        
    Returns:
        street_multiplier: Factor de ajuste
    """
    street_multipliers = jnp.array([
        1.0,  # Preflop (index 0, no usado)
        1.0,  # (index 1, no usado)  
        1.0,  # (index 2, no usado)
        1.1,  # Flop (index 3) - Más agresivo con draws
        0.9,  # Turn (index 4) - Más cauteloso
        0.8   # River (index 5) - Muy cauteloso, showdown
    ])
    
    safe_index = jnp.clip(num_community_cards, 0, 5)
    return street_multipliers[safe_index]

@jax.jit
def analyze_hand_vs_board(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Analiza qué tan bien conecta nuestra mano con el board.
    100% compatible con JAX JIT.
    """
    num_community = jnp.sum(community_cards >= 0)
    
    # Analizar cartas combinadas
    all_rank_counts = jnp.zeros(13, dtype=jnp.int32)
    all_suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    # Hole cards
    for i in range(2):
        rank = hole_cards[i] // 4
        suit = hole_cards[i] % 4
        all_rank_counts = all_rank_counts.at[rank].add(1)
        all_suit_counts = all_suit_counts.at[suit].add(1)
    
    # Community cards
    max_community = jnp.minimum(num_community, 5)
    for i in range(5):
        valid_card = jnp.where(i < max_community, community_cards[i], -1)
        rank = jnp.where(valid_card >= 0, valid_card // 4, 0)
        suit = jnp.where(valid_card >= 0, valid_card % 4, 0)
        
        all_rank_counts = jnp.where(
            valid_card >= 0,
            all_rank_counts.at[rank].add(1),
            all_rank_counts
        )
        all_suit_counts = jnp.where(
            valid_card >= 0,
            all_suit_counts.at[suit].add(1),
            all_suit_counts
        )
    
    # Evaluar fuerza
    max_rank_count = jnp.max(all_rank_counts)
    max_suit_count = jnp.max(all_suit_counts)
    pairs_count = jnp.sum(all_rank_counts >= 2)
    
    made_hand_strength = jnp.where(
        max_rank_count >= 4, 1.0,
        jnp.where(
            (max_rank_count >= 3) & (pairs_count >= 2), 0.95,
            jnp.where(
                max_suit_count >= 5, 0.9,
                jnp.where(
                    max_rank_count >= 3, 0.7,
                    jnp.where(
                        pairs_count >= 2, 0.5,
                        jnp.where(max_rank_count >= 2, 0.3, 0.1)
                    )
                )
            )
        )
    )
    
    # DRAWS
    hole_suits = hole_cards % 4
    suited_hole = hole_suits[0] == hole_suits[1]
    our_suit_count = jnp.where(
        suited_hole,
        all_suit_counts[hole_suits[0]],
        0
    )
    
    # Solo en flop/turn (no river)
    draw_strength = jnp.where(
        num_community < 5,
        jnp.where(
            our_suit_count == 4, 0.4,
            jnp.where(our_suit_count == 3, 0.2, 0.0)
        ),
        0.0
    )
    
    final_strength = jnp.clip(made_hand_strength + draw_strength, 0.0, 1.0)
    
    # Si preflop, retornar neutral
    return jnp.where(num_community >= 3, final_strength, 0.5) 