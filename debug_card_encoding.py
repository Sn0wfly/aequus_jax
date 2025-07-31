import jax.numpy as jnp

def debug_card_encoding():
    """Debug card encoding to find correct AA representation."""
    
    # Standard card encoding: 0-51
    # Suits: 0=spades, 1=hearts, 2=diamonds, 3=clubs
    # Ranks: 0=2, 1=3, ..., 12=A
    
    print("Card encoding test:")
    
    # Test different AA representations
    test_cards = [
        [51, 47],  # Current test
        [48, 44],  # Alternative
        [12, 13],  # Another possibility
        [0, 1],    # Another possibility
    ]
    
    for i, cards in enumerate(test_cards):
        ranks = jnp.array(cards) // 4
        suits = jnp.array(cards) % 4
        
        print(f"\nTest {i+1}: Cards {cards}")
        print(f"  Ranks: {ranks}")
        print(f"  Suits: {suits}")
        print(f"  Is pair: {ranks[0] == ranks[1]}")
        print(f"  Is suited: {suits[0] == suits[1]}")
        
        if ranks[0] == ranks[1] and ranks[0] == 12:
            print(f"  *** This is AA! ***")
        elif ranks[0] == ranks[1]:
            print(f"  *** This is a pair of rank {ranks[0]} ***")
    
    # Find the correct AA representation
    print("\nSearching for AA...")
    for card1 in range(52):
        for card2 in range(52):
            if card1 != card2:
                ranks = jnp.array([card1, card2]) // 4
                if ranks[0] == ranks[1] and ranks[0] == 12:
                    suits = jnp.array([card1, card2]) % 4
                    print(f"AA found: cards [{card1}, {card2}], ranks {ranks}, suits {suits}")

if __name__ == "__main__":
    debug_card_encoding() 