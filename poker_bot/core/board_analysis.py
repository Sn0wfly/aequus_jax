import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def analyze_board_texture(community_cards: jnp.ndarray) -> jnp.ndarray:
    """
    Analiza la textura del board: "wet" (con proyectos) vs "dry" (estático).
    Retorna un valor entre 0.0 (muy dry) y 1.0 (muy wet).
    """
    num_cards = jnp.sum(community_cards >= 0).astype(jnp.int32)
    
    # Si es preflop o no hay suficientes cartas en el flop, la textura es neutral.
    is_insufficient = num_cards < 3
    
    def calculate_texture():
        # Tomar solo las cartas válidas del board
        valid_cards = community_cards
        
        # --- Conteo de proyectos de color (Flush Draws) ---
        suit_counts = jnp.zeros(4, dtype=jnp.int32)
        for i in range(5):
             card_is_valid = valid_cards[i] >= 0
             suit = valid_cards[i] % 4
             suit_counts = suit_counts.at[suit].add(jnp.where(card_is_valid, 1, 0))
        
        max_suit_count = jnp.max(suit_counts)
        flush_draw_strength = jnp.where(max_suit_count >= 3, 0.5, jnp.where(max_suit_count == 2, 0.2, 0.0))

        # --- Conteo de proyectos de escalera (Straight Draws) ---
        ranks = jnp.where(valid_cards >= 0, valid_cards // 4, -1)
        unique_ranks = jnp.unique(ranks, size=5, fill_value=-1)
        sorted_ranks = jnp.sort(unique_ranks)
        
        # Eliminar valores de relleno para un cálculo de gaps correcto
        valid_sorted_ranks = jnp.where(sorted_ranks >= 0, sorted_ranks, jnp.nan)
        gaps = jnp.diff(valid_sorted_ranks)
        
        # Contar cuántas cartas están a 1 o 2 de distancia (conectividad)
        connectivity = jnp.sum(jnp.where((gaps > 0) & (gaps < 4), 1, 0))
        straight_draw_strength = jnp.clip(connectivity / 4.0, 0.0, 0.5)

        # --- Conteo de cartas altas y si el board está pareado ---
        rank_counts = jnp.zeros(13, dtype=jnp.int32)
        for i in range(5):
            card_is_valid = valid_cards[i] >= 0
            rank = valid_cards[i] // 4
            rank_counts = rank_counts.at[rank].add(jnp.where(card_is_valid, 1, 0))

        is_paired = jnp.any(rank_counts > 1)
        high_card_factor = jnp.sum(rank_counts[9:]) / 5.0 # T, J, Q, K, A
        
        # Combinar todos los factores para obtener la "humedad" total del board
        total_wetness = flush_draw_strength + straight_draw_strength + (0.2 * is_paired) + (0.1 * high_card_factor)
        return jnp.clip(total_wetness, 0.0, 1.0)

    # Usa lax.cond para ser compatible con la compilación JIT
    return lax.cond(
        is_insufficient,
        lambda: 0.5,  # Valor neutral si no hay suficientes cartas
        calculate_texture
    )

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
                    max_rank_count >= 3, 0.8,  # Trips (aumentado para sets)
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