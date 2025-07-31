import jax.numpy as jnp
from poker_bot.core.board_analysis import analyze_hand_vs_board

def debug_set_detection():
    """Debug why JJ on AKQ isn't detected as a set."""
    
    # Test JJ on AKQ
    jj = jnp.array([39, 38])  # JJ (ranks 9, 9)
    akq_board = jnp.array([48, 44, 40, -1, -1])  # AKQ (ranks 12, 11, 10)
    
    print("JJ cards:", jj)
    print("JJ ranks:", jj // 4)
    print("AKQ board:", akq_board)
    print("AKQ board ranks:", akq_board // 4)
    
    # Manual calculation
    all_cards = jnp.concatenate([jj, akq_board])
    all_ranks = all_cards // 4
    print("All cards:", all_cards)
    print("All ranks:", all_ranks)
    
    # Count ranks
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(7):
        if all_cards[i] >= 0:
            rank = all_cards[i] // 4
            rank_counts = rank_counts.at[rank].add(1)
    
    print("Rank counts:", rank_counts)
    print("Max rank count:", jnp.max(rank_counts))
    print("Pairs count:", jnp.sum(rank_counts >= 2))
    
    # Test the function
    strength = analyze_hand_vs_board(jj, akq_board)
    print("Function result:", strength)

if __name__ == "__main__":
    debug_set_detection() 