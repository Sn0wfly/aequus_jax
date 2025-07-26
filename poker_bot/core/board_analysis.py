import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def analyze_board_texture(community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Analiza textura del board: wet (draws) vs dry (estático).
    
    Args:
        community_cards: Array de 5 cartas community (-1 si no hay)
        
    Returns:
        board_wetness: float 0.0 (dry) - 1.0 (wet)
    """
    # Filtrar cartas válidas
    valid_cards = community_cards[community_cards >= 0]
    num_cards = jnp.sum(community_cards >= 0)
    
    # Si no hay flop, retornar neutral
    if num_cards < 3:
        return 0.5
        
    ranks = valid_cards // 4  # 0-12
    suits = valid_cards % 4   # 0-3
    
    # FLUSH DRAWS
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(jnp.minimum(num_cards, 5)):
        suit_counts = suit_counts.at[suits[i]].add(1)
    
    max_suit_count = jnp.max(suit_counts)
    flush_draw_strength = jnp.where(
        max_suit_count >= 3, 0.4,  # 3+ mismo palo = flush draw
        jnp.where(max_suit_count == 2, 0.2, 0.0)  # 2 mismo palo = backdoor
    )
    
    # STRAIGHT DRAWS
    rank_mask = jnp.zeros(13, dtype=jnp.bool_)
    for i in range(jnp.minimum(num_cards, 5)):
        rank_mask = rank_mask.at[ranks[i]].set(True)
    
    # Buscar secuencias
    straight_potential = 0.0
    for start_rank in range(10):  # A-2-3-4-5 hasta T-J-Q-K-A
        sequence = rank_mask[start_rank:start_rank+3]
        if jnp.sum(sequence) >= 2:  # 2+ cartas consecutivas
            straight_potential = jnp.maximum(straight_potential, 0.3)
            
    # PAIRED BOARDS
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(jnp.minimum(num_cards, 5)):
        rank_counts = rank_counts.at[ranks[i]].add(1)
        
    max_rank_count = jnp.max(rank_counts)
    pairs_count = jnp.sum(rank_counts >= 2)
    
    pair_texture = jnp.where(
        max_rank_count >= 3, 0.8,  # Trips/quads en board
        jnp.where(pairs_count >= 2, 0.6,  # Two pair en board
                  jnp.where(max_rank_count == 2, 0.3, 0.0))  # Un par
    )
    
    # HIGH CARDS (broadway)
    high_cards = jnp.sum(ranks >= 9)  # T, J, Q, K, A
    high_card_factor = high_cards / num_cards * 0.2
    
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
    
    Returns:
        connection_strength: 0.0 (no connection) - 1.0 (nuts)
    """
    valid_community = community_cards[community_cards >= 0]
    num_community = jnp.sum(community_cards >= 0)
    
    if num_community < 3:  # Preflop
        return 0.5
        
    # Combinar todas las cartas para análisis
    all_cards = jnp.concatenate([hole_cards, valid_community])
    all_ranks = all_cards // 4
    all_suits = all_cards % 4
    
    # MADE HANDS
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    for i in range(len(all_cards)):
        rank_counts = rank_counts.at[all_ranks[i]].add(1)
        suit_counts = suit_counts.at[all_suits[i]].add(1)
    
    # Evaluar fuerza de mano hecha
    max_rank_count = jnp.max(rank_counts)
    max_suit_count = jnp.max(suit_counts)
    pairs_count = jnp.sum(rank_counts >= 2)
    
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
        # Flush draws
        hole_suits = hole_cards % 4
        if hole_suits[0] == hole_suits[1]:  # Suited hole cards
            our_suit_count = jnp.sum(all_suits == hole_suits[0])
            draw_strength = jnp.where(
                our_suit_count == 4, 0.4,  # Flush draw
                jnp.where(our_suit_count == 3, 0.2, 0.0)  # Backdoor
            )
    
    return jnp.clip(made_hand_strength + draw_strength, 0.0, 1.0) 