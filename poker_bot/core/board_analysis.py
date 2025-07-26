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
    
    # STRAIGHT DRAWS - versión 100% JAX compatible
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(5):
        valid_card = jnp.where(i < max_cards, community_cards[i], -1)
        rank = jnp.where(valid_card >= 0, valid_card // 4, 0)
        rank_counts = jnp.where(
            valid_card >= 0,
            rank_counts.at[rank].add(1),
            rank_counts
        )

    # Detectar secuencias de forma simple y compatible
    straight_potential = 0.0

    # Check for regular straights (no wheel)
    for start_rank in range(9):  # 0-8 (2-3-4 hasta T-J-Q)
        consecutive_cards = 0
        for offset in range(3):  # Ventana de 3 cartas
            if start_rank + offset < 13:
                has_card = rank_counts[start_rank + offset] > 0
                consecutive_cards += jnp.where(has_card, 1, 0)
        
        # Si hay 2+ cartas consecutivas, hay straight potential
        has_potential = consecutive_cards >= 2
        straight_potential = jnp.maximum(
            straight_potential, 
            jnp.where(has_potential, 0.3, 0.0)
        )

    # Wheel straight (A-2-3-4-5) - versión JAX compatible
    ace_count = rank_counts[12]  # A
    two_count = rank_counts[0]   # 2
    three_count = rank_counts[1] # 3
    four_count = rank_counts[2]  # 4
    five_count = rank_counts[3]  # 5

    wheel_cards = (
        jnp.where(ace_count > 0, 1, 0) +
        jnp.where(two_count > 0, 1, 0) +
        jnp.where(three_count > 0, 1, 0) +
        jnp.where(four_count > 0, 1, 0) +
        jnp.where(five_count > 0, 1, 0)
    )

    has_wheel_potential = wheel_cards >= 2
    straight_potential = jnp.maximum(
        straight_potential,
        jnp.where(has_wheel_potential, 0.3, 0.0)
    )

    # PAIRED BOARDS - versión menos agresiva
    max_rank_count = jnp.max(rank_counts)
    pairs_count = jnp.sum(rank_counts >= 2)

    pair_texture = jnp.where(
        max_rank_count >= 3, 0.6,  # Trips/quads (reducido de 0.8)
        jnp.where(pairs_count >= 2, 0.4,  # Two pair (reducido de 0.6)
                  jnp.where(max_rank_count == 2, 0.2, 0.0))  # Un par (reducido de 0.3)
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
    100% compatible con JAX JIT - sin dynamic slicing.
    """
    num_community = jnp.sum(community_cards >= 0)
    
    # Analizar cartas combinadas
    all_rank_counts = jnp.zeros(13, dtype=jnp.int32)
    all_suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    # Hole cards
    hole_ranks = hole_cards // 4
    hole_suits = hole_cards % 4
    
    for i in range(2):
        rank = hole_ranks[i]
        suit = hole_suits[i]
        all_rank_counts = all_rank_counts.at[rank].add(1)
        all_suit_counts = all_suit_counts.at[suit].add(1)
    
    # Community cards + track highest board rank
    max_community = jnp.minimum(num_community, 5)
    highest_board_rank = 0
    
    for i in range(5):
        valid_card = jnp.where(i < max_community, community_cards[i], -1)
        rank = jnp.where(valid_card >= 0, valid_card // 4, 0)
        suit = jnp.where(valid_card >= 0, valid_card % 4, 0)
        
        # Track highest board rank (compatible with JIT)
        highest_board_rank = jnp.where(
            (valid_card >= 0) & (rank > highest_board_rank),
            rank,
            highest_board_rank
        )
        
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
    
    # Evaluar fuerza base
    max_rank_count = jnp.max(all_rank_counts)
    max_suit_count = jnp.max(all_suit_counts)
    pairs_count = jnp.sum(all_rank_counts >= 2)
    
    # Detectar overpair
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]
    is_overpair = is_pocket_pair & (hole_ranks[0] > highest_board_rank) & (num_community >= 3)
    
    # Scoring con overpair detection
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
                        jnp.where(
                            is_overpair, 0.75,  # *** OVERPAIR FUERTE ***
                            jnp.where(
                                max_rank_count >= 2, 0.3,  # Regular pair
                                0.1  # High card
                            )
                        )
                    )
                )
            )
        )
    )
    
    # DRAWS
    suited_hole = hole_suits[0] == hole_suits[1]
    our_suit_count = jnp.where(
        suited_hole,
        all_suit_counts[hole_suits[0]],
        0
    )
    
    draw_strength = jnp.where(
        num_community < 5,
        jnp.where(
            our_suit_count == 4, 0.4,
            jnp.where(our_suit_count == 3, 0.2, 0.0)
        ),
        0.0
    )
    
    final_strength = jnp.clip(made_hand_strength + draw_strength, 0.0, 1.0)
    
    return jnp.where(num_community >= 3, final_strength, 0.5) 