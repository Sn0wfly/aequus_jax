import jax.numpy as jnp

def debug_hand_classification():
    """Debug hand classification logic."""
    # Test AA
    aa_cards = jnp.array([51, 47])  # AA
    ranks = aa_cards // 4
    suits = aa_cards % 4
    
    print(f"AA cards: {aa_cards}")
    print(f"AA ranks: {ranks}")
    print(f"AA suits: {suits}")
    
    high_rank = jnp.max(ranks)
    low_rank = jnp.min(ranks)
    is_suited = suits[0] == suits[1]
    is_pair = ranks[0] == ranks[1]
    
    print(f"High rank: {high_rank}")
    print(f"Low rank: {low_rank}")
    print(f"Is suited: {is_suited}")
    print(f"Is pair: {is_pair}")
    
    # Test premium conditions
    premium_pairs = (is_pair & (high_rank >= 10))
    premium_unpaired = (
        ((high_rank == 12) & (low_rank >= 10)) |  # AK, AQ, AJ  
        ((high_rank == 11) & (low_rank >= 10))    # KQ, KJ
    )
    
    print(f"Premium pairs condition: {premium_pairs}")
    print(f"Premium unpaired condition: {premium_unpaired}")
    
    # Test JJ
    jj_cards = jnp.array([39, 38])  # JJ
    jj_ranks = jj_cards // 4
    jj_suits = jj_cards % 4
    
    print(f"\nJJ cards: {jj_cards}")
    print(f"JJ ranks: {jj_ranks}")
    
    jj_high_rank = jnp.max(jj_ranks)
    jj_low_rank = jnp.min(jj_ranks)
    jj_is_pair = jj_ranks[0] == jj_ranks[1]
    
    print(f"JJ high rank: {jj_high_rank}")
    print(f"JJ low rank: {jj_low_rank}")
    print(f"JJ is pair: {jj_is_pair}")
    
    jj_premium_pairs = (jj_is_pair & (jj_high_rank >= 10))
    print(f"JJ premium pairs condition: {jj_premium_pairs}")

if __name__ == "__main__":
    debug_hand_classification() 