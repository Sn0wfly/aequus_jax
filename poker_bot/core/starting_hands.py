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
    premium_pairs = (is_pair & (high_rank >= 10))  # AA, KK, QQ, JJ
    premium_unpaired = (
        ((high_rank == 12) & (low_rank >= 10)) |  # AK, AQ, AJ  
        ((high_rank == 11) & (low_rank >= 10))    # KQ, KJ
    )
    
    # STRONG HANDS (0.65-0.84)
    strong_pairs = (is_pair & (high_rank >= 6) & (high_rank <= 9))  # 77-TT
    strong_aces = ((high_rank == 12) & (low_rank >= 7) & (low_rank <= 9))  # AT-A9
    strong_suited = (is_suited & ((high_rank >= 9) & (low_rank >= 6)))  # Suited broadway
    
    # MEDIUM HANDS (0.35-0.64)
    medium_pairs = (is_pair & (high_rank >= 2) & (high_rank <= 5))  # 22-66
    medium_aces = ((high_rank == 12) & (low_rank >= 2) & (low_rank <= 6))  # A2-A8
    suited_connectors = (is_suited & (high_rank - low_rank <= 4) & (low_rank >= 5))  # 65s+
    
    # WEAK HANDS (0.1-0.34)
    # Todo lo demás
    
    return jnp.where(
        premium_pairs, 0.95,
        jnp.where(
            premium_unpaired & is_suited, 0.90,
            jnp.where(
                premium_unpaired, 0.85,
                jnp.where(
                    strong_pairs, 0.75,
                    jnp.where(
                        strong_aces & is_suited, 0.70,
                        jnp.where(
                            strong_suited, 0.68,
                            jnp.where(
                                medium_pairs, 0.50,
                                jnp.where(
                                    suited_connectors, 0.45,
                                    jnp.where(
                                        medium_aces, 0.35,
                                        0.15  # Trash hands
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
    0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
    """
    position_values = jnp.array([0.8, 0.85, 0.95, 1.1, 0.9, 0.85])
    return position_values[position] 