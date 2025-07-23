# Debug profundo del training loop para encontrar el bug
import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import _update_regrets_for_game_pure, _compute_real_cfr_regrets
from poker_bot.core.bucketing import compute_info_set_id
from poker_bot.core import full_game_engine as game_engine

print("🔍 DEEP TRAINING LOOP DEBUG:")
print("=" * 50)

# Simular exactamente una iteración de training
batch_size = 1  # Reducir para debug simple
regrets = jnp.zeros((50000, 9))
strategy = jnp.ones((50000, 9)) / 9

print("🎮 Simulando 1 juego de training...")

# Generate keys para 1 juego
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, batch_size)

# Simular game engine results
print("📊 Simulando resultados del game engine...")
mock_game_results = {
    'hole_cards': jnp.array([[29, 6], [51, 47], [23, 0], [44, 45], [12, 13], [8, 9]]),  # 6 players
    'final_community': jnp.array([34, 11, 3, 19, 41]),
    'final_pot': jnp.array([100.0]),
    'player_stacks': jnp.zeros(6),
    'player_bets': jnp.zeros(6)
}

mock_payoffs = jnp.array([50.0, -10.0, -10.0, -10.0, -10.0, -10.0])  # Player 0 wins

print(f"  Player 0 hole cards: {mock_game_results['hole_cards'][0]} (mano débil)")
print(f"  Community: {mock_game_results['final_community']}")
print(f"  Payoffs: {mock_payoffs}")

# Test direct regret computation para player 0
print("\n🧮 Computando regrets directamente para Player 0...")
hole_cards = mock_game_results['hole_cards'][0]
community_cards = mock_game_results['final_community']
pot_size = jnp.array([100.0])

action_regrets = _compute_real_cfr_regrets(
    hole_cards, community_cards, 0, pot_size, mock_payoffs, strategy, 9
)

print(f"📊 Action regrets computados:")
actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
for action, regret in zip(actions, action_regrets):
    print(f"  {action:12}: {regret:8.3f}")

# Compute info set ID
info_set_id = compute_info_set_id(hole_cards, community_cards, 0, pot_size)
print(f"\n🎯 Info Set ID: {info_set_id}")

# Test regret update function completa
print(f"\n🔧 Testing regret update function...")
game_key = jax.random.PRNGKey(123)

regret_updates = _update_regrets_for_game_pure(
    regrets, strategy, mock_game_results, mock_payoffs, 9, game_key
)

print(f"📈 Regret updates para info set {info_set_id}:")
for action, update in zip(actions, regret_updates[info_set_id]):
    print(f"  {action:12}: {update:8.6f}")

# Verificar si los updates hacen sentido
print(f"\n🚨 Verificación de consistencia:")
expected_fold = action_regrets[0] * 0.02  # learning_rate = 0.02
actual_fold = regret_updates[info_set_id][0]

print(f"  Expected FOLD update: {expected_fold:.6f}")
print(f"  Actual FOLD update: {actual_fold:.6f}")
print(f"  Ratio: {actual_fold/expected_fold if expected_fold != 0 else 'N/A':.3f}")

if abs(actual_fold - expected_fold) < 0.001:
    print(f"  ✅ Updates están correctos!")
else:
    print(f"  ❌ Updates están incorrectos!")
    print(f"      Gap: {abs(actual_fold - expected_fold):.6f}")

# Check MC sampling
print(f"\n🎲 MC Sampling analysis:")
from poker_bot.core.mccfr_algorithm import mc_sampling_strategy

info_set_indices = jnp.array([info_set_id])
sampling_mask = mc_sampling_strategy(regrets, info_set_indices, game_key)
print(f"  Sampling mask for info set {info_set_id}: {sampling_mask[0]}")
print(f"  MC sampling rate: 1.0 (debería ser True siempre)")

if not sampling_mask[0]:
    print(f"  ❌ Info set NO fue sampled!")
else:
    print(f"  ✅ Info set fue sampled correctamente")