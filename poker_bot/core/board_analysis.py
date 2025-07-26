import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def analyze_board_texture(community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Analiza textura del board: wet (draws) vs dry (estático).
    Compatible con JAX JIT - no usa boolean indexing.
    """
    # Contar cartas válidas
    num_cards = jnp.sum(community_cards >= 0)
    
    # Si no hay flop, retornar neutral
    if num_cards < 3:
        return 0.5
        
    # Procesar solo las primeras num_cards (máximo 5)
    max_cards = jnp.minimum(num_cards, 5)
    
    # FLUSH DRAWS - versión compatible con JIT
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(5):  # Máximo 5 community cards
        valid_card = jnp.where(i < max_cards, community_cards[i], -1)
        suit = jnp.where(valid_card >= 0, valid_card % 4, 0)
        suit_counts = jnp.where(
            valid_card >= 0,
            suit_counts.at[suit].add(1),
            suit_counts
        )
    
    max_suit_count = jnp.max(suit_counts)
    flush_draw_strength = jnp.where(
        max_suit_count >= 3, 0.4,  # 3+ mismo palo = flush draw
        jnp.where(max_suit_count == 2, 0.2, 0.0)  # 2 mismo palo = backdoor
    )
    
    # STRAIGHT DRAWS - versión compatible con JIT
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(5):
        valid_card = jnp.where(i < max_cards, community_cards[i], -1)
        rank = jnp.where(valid_card >= 0, valid_card // 4, 0)
        rank_counts = jnp.where(
            valid_card >= 0,
            rank_counts.at[rank].add(1),
            rank_counts
        )
    
    # Detectar secuencias (simplificado)
    consecutive_ranks = 0
    for start_rank in range(11):  # 0-10 para evitar overflow
        has_rank = rank_counts[start_rank] > 0
        has_next = rank_counts[start_rank + 1] > 0
        has_after = rank_counts[start_rank + 2] > 0
        if has_rank & has_next:
            consecutive_ranks = jnp.maximum(consecutive_ranks, 2)
        if has_rank & has_next & has_after:
            consecutive_ranks = jnp.maximum(consecutive_ranks, 3)
    
    straight_potential = jnp.where(consecutive_ranks >= 2, 0.3, 0.0)
            
    # PAIRED BOARDS
    max_rank_count = jnp.max(rank_counts)
    pairs_count = jnp.sum(rank_counts >= 2)
    
    pair_texture = jnp.where(
        max_rank_count >= 3, 0.8,  # Trips/quads en board
        jnp.where(pairs_count >= 2, 0.6,  # Two pair en board
                  jnp.where(max_rank_count == 2, 0.3, 0.0))  # Un par
    )
    
    # HIGH CARDS (broadway) - ranks 9-12 (T, J, Q, K, A)
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
    
    return jnp.clip(total_wetness, 0.0, 1.0)

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
    Compatible con JAX JIT.
    """
    num_community = jnp.sum(community_cards >= 0)
    
    if num_community < 3:  # Preflop
        return 0.5
        
    # Analizar cartas combinadas (hole + community)
    all_rank_counts = jnp.zeros(13, dtype=jnp.int32)
    all_suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    # Procesar hole cards
    for i in range(2):
        rank = hole_cards[i] // 4
        suit = hole_cards[i] % 4
        all_rank_counts = all_rank_counts.at[rank].add(1)
        all_suit_counts = all_suit_counts.at[suit].add(1)
    
    # Procesar community cards
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
    
    # Evaluar fuerza de mano hecha
    max_rank_count = jnp.max(all_rank_counts)
    max_suit_count = jnp.max(all_suit_counts)
    pairs_count = jnp.sum(all_rank_counts >= 2)
    
    made_hand_strength = jnp.where(
        max_rank_count >= 4, 1.0,  # Quads
        jnp.where(
            (max_rank_count >= 3) & (pairs_count >= 2), 0.95,  # Full house
            jnp.where(
                max_suit_count >= 5, 0.9,  # Flush
                jnp.where(
                    max_rank_count >= 3, 0.7,  # Trips
                    jnp.where(
                        pairs_count >= 2, 0.5,  # Two pair
                        jnp.where(max_rank_count >= 2, 0.3, 0.1)  # One pair / high card
                    )
                )
            )
        )
    )
    
    # DRAWS (solo en flop/turn)
    draw_strength = 0.0
    if num_community < 5:  # No draws en river
        # Flush draws - simplificado
        hole_suits = hole_cards % 4
        suited_hole = hole_suits[0] == hole_suits[1]
        our_suit_count = jnp.where(
            suited_hole,
            all_suit_counts[hole_suits[0]],
            0
        )
        draw_strength = jnp.where(
            our_suit_count == 4, 0.4,  # Flush draw
            jnp.where(our_suit_count == 3, 0.2, 0.0)  # Backdoor
        )
    
    return jnp.clip(made_hand_strength + draw_strength, 0.0, 1.0) 