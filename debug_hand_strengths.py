import jax.numpy as jnp
from poker_bot.core.starting_hands import classify_starting_hand

def debug_hand_strengths():
    """Debug exact hand strengths for pocket pairs."""
    
    test_hands = [
        ([48, 49], "AA"),
        ([44, 45], "KK"),
        ([40, 41], "QQ"),
        ([39, 38], "JJ"),
        ([35, 34], "TT"),
        ([31, 30], "99"),
        ([27, 26], "88"),
        ([23, 22], "77"),
    ]
    
    print("Pocket pair hand strengths:")
    print("=" * 30)
    
    for hole_cards, name in test_hands:
        strength = classify_starting_hand(jnp.array(hole_cards))
        print(f"{name}: {strength:.3f}")
    
    print(f"\nPremium threshold: >=0.70")
    print("JJ should be premium, TT/99 should not be")

if __name__ == "__main__":
    debug_hand_strengths() 