import jax.numpy as jnp
from poker_bot.core.starting_hands import classify_starting_hand

def debug_premium_threshold():
    """Debug why JJ, TT, 99 are being treated as premium hands."""
    
    test_hands = [
        ([48, 49], "AA"),
        ([44, 45], "KK"),
        ([40, 41], "QQ"),
        ([39, 38], "JJ"),
        ([35, 34], "TT"),
        ([31, 30], "99"),
        ([48, 44], "AK"),
        ([48, 40], "AQ"),
        ([48, 36], "AJ"),
        ([44, 40], "KQ"),
    ]
    
    print("Hand strength analysis:")
    print("=" * 30)
    
    for hole_cards, name in test_hands:
        strength = classify_starting_hand(jnp.array(hole_cards))
        is_premium = strength > 0.7
        print(f"{name:4s}: {strength:.3f} {'(PREMIUM)' if is_premium else '(NOT PREMIUM)'}")
    
    print(f"\nPremium threshold: >0.7")
    print("Only AA, KK, QQ, AK, AQ, KQ should be premium")

if __name__ == "__main__":
    debug_premium_threshold() 