# test_position_awareness.py
import jax.numpy as jnp
from poker_bot.core.starting_hands import classify_starting_hand_with_position

# Test A9o en diferentes posiciones
a9o = jnp.array([48, 32])  # A♠ 9♣

print("=== POSITION AWARENESS TEST ===")
positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

for i, pos_name in enumerate(positions):
    strength = classify_starting_hand_with_position(a9o, i)
    print(f"{pos_name}: A9o strength = {strength:.3f}")

print("\n✅ Esperado: UTG < MP < CO < BTN")