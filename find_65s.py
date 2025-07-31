import jax.numpy as jnp

def find_65s():
    """Find the correct card encoding for 65s."""
    
    # 6 = rank 4, 5 = rank 3
    # We need cards with ranks 4 and 3, same suit
    
    print("Searching for 65s...")
    for card1 in range(52):
        for card2 in range(52):
            if card1 != card2:
                ranks = jnp.array([card1, card2]) // 4
                suits = jnp.array([card1, card2]) % 4
                
                # Check if it's 65s (ranks 4,3, same suit)
                if (ranks[0] == 4 and ranks[1] == 3 and suits[0] == suits[1]) or \
                   (ranks[0] == 3 and ranks[1] == 4 and suits[0] == suits[1]):
                    print(f"65s found: cards [{card1}, {card2}], ranks {ranks}, suits {suits}")
                
                # Also check for 76s (ranks 5,4, same suit)
                if (ranks[0] == 5 and ranks[1] == 4 and suits[0] == suits[1]) or \
                   (ranks[0] == 4 and ranks[1] == 5 and suits[0] == suits[1]):
                    print(f"76s found: cards [{card1}, {card2}], ranks {ranks}, suits {suits}")

if __name__ == "__main__":
    find_65s() 