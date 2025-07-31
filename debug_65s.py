import jax.numpy as jnp
from poker_bot.core.starting_hands import classify_starting_hand

def debug_65s():
    """Debug why 65s is not being classified as suited connector."""
    
    # Test 65s
    cards = jnp.array([23, 19])  # 65s
    ranks = cards // 4
    suits = cards % 4
    
    print("65s cards:", cards)
    print("65s ranks:", ranks)
    print("65s suits:", suits)
    
    high_rank = jnp.max(ranks)
    low_rank = jnp.min(ranks)
    is_suited = suits[0] == suits[1]
    is_pair = ranks[0] == ranks[1]
    
    print(f"High rank: {high_rank}")
    print(f"Low rank: {low_rank}")
    print(f"Is suited: {is_suited}")
    print(f"Is pair: {is_pair}")
    print(f"Rank difference: {high_rank - low_rank}")
    
    # Test suited connector conditions
    suited_connectors = (is_suited & (high_rank - low_rank <= 4) & (low_rank >= 5))
    print(f"Suited connector condition: {suited_connectors}")
    
    # Test the function
    strength = classify_starting_hand(cards)
    print(f"Function result: {strength}")

if __name__ == "__main__":
    debug_65s() 