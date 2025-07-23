# Debug directo de action_values para verificar que se estén aplicando
import jax.numpy as jnp
from poker_bot.core.trainer import _compute_real_cfr_regrets, _evaluate_7card_simple
import numpy as np

print("🔍 TESTING ACTION VALUES DIRECTLY:")
print("=" * 50)

# Simular datos de entrada
hole_cards = jnp.array([29, 6])  # Mano débil del test
community_cards = jnp.array([34, 11, 3, 19, 41])
player_idx = 0
pot_size = jnp.array([50.0])
game_payoffs = jnp.zeros(6)  # Mock payoffs
strategy = jnp.ones((200000, 9)) / 9  # Strategy uniforme inicial
num_actions = 9

print(f"Input mano: {hole_cards}")
print(f"Community: {community_cards}")

# Llamar directamente a la función que computa regrets
try:
    action_regrets = _compute_real_cfr_regrets(
        hole_cards, community_cards, player_idx, pot_size, 
        game_payoffs, strategy, num_actions
    )
    
    print(f"\n📊 Action Values Computados:")
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    for i, (action, regret) in enumerate(zip(actions, action_regrets)):
        print(f"  {action:12}: {regret:8.3f}")
    
    print(f"\n🎯 Expectativa para mano débil (strength ~0.067):")
    print(f"  FOLD debería ser ALTO (~+1.8)")
    print(f"  BET_LARGE debería ser BAJO (~-0.6)")
    
    fold_value = action_regrets[0]
    bet_large_value = action_regrets[5]
    
    print(f"\n🚨 Verificación:")
    print(f"  FOLD: {fold_value:.3f}")
    print(f"  BET_LARGE: {bet_large_value:.3f}")
    
    if fold_value > bet_large_value and fold_value > 0.5:
        print(f"  ✅ Action values están CORRECTOS")
    else:
        print(f"  ❌ Action values están INCORRECTOS!")
        print(f"      FOLD debería > BET_LARGE y > 0.5")

except Exception as e:
    print(f"❌ Error al computar action values: {e}")
    import traceback
    traceback.print_exc()


print("\n�� TESTING FLUSH HAND (Game 29):")
print("=" * 40)

# Game 29 exacto que falló
flush_cards = jnp.array([37, 13])  # Cards from failing game
flush_community = jnp.array([41, 31, 26, 49, 5])  # Community from failing game

# Test hand strength evaluation
try:
    strength = _evaluate_7card_simple(flush_cards, flush_community)
    print(f"Game 29 strength: {strength:.3f} (should be 0.750)")
    
    # Test action values para flush
    flush_regrets = _compute_real_cfr_regrets(
        flush_cards, flush_community, 0, jnp.array([50.0]), 
        jnp.zeros(6), jnp.ones((50000, 9)) / 9, 9
    )
    
    print(f"\n📊 Action Values para FLUSH:")
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    for i, (action, regret) in enumerate(zip(actions, flush_regrets)):
        print(f"  {action:12}: {regret:8.3f}")
    
    fold_regret = flush_regrets[0]
    allin_regret = flush_regrets[8]
    
    print(f"\n🚨 Flush Verification:")
    print(f"  FOLD: {fold_regret:.3f} (should be NEGATIVE)")
    print(f"  ALL_IN: {allin_regret:.3f} (should be POSITIVE)")
    
    if fold_regret < 0 and allin_regret > 0:
        print(f"  ✅ Flush action values CORRECT")
    else:
        print(f"  ❌ Flush action values BROKEN!")
        print(f"      This explains the fold-flush bug!")

except Exception as e:
    print(f"❌ Error testing flush: {e}")
    import traceback
    traceback.print_exc()


print("\n🔍 TESTING INFO SET COLLISION:")
print("=" * 40)

from poker_bot.core.bucketing import compute_info_set_id

# Test collision between flush y mano débil
weak_cards = jnp.array([29, 6])  # Mano débil
flush_cards = jnp.array([37, 13])  # Flush problemático  
community = jnp.array([41, 31, 26, 49, 5])
pot = jnp.array([50.0])

weak_info_set = compute_info_set_id(weak_cards, community, 0, pot)
flush_info_set = compute_info_set_id(flush_cards, community, 0, pot)

print(f"Weak hand info set:  {weak_info_set}")
print(f"Flush hand info set: {flush_info_set}")

if weak_info_set == flush_info_set:
    print("❌ COLLISION DETECTED! Same info set for different hands!")
    print("   This explains the fold-flush bug!")
else:
    print("✅ No collision - different info sets")
    print("   Bug must be elsewhere...")


print("\n🔍 TESTING STORED STRATEGY:")
print("=" * 40)

# Cargar modelo entrenado
try:
    import pickle
    with open('models/trainerfix.pkl', 'rb') as f:
        model = pickle.load(f)
    
    strategy = model['strategy']
    regrets = model['regrets']
    
    # Estrategia para flush info set
    flush_info_set = 4015
    flush_strategy = strategy[flush_info_set]
    flush_regrets = regrets[flush_info_set]
    
    print(f"Info Set {flush_info_set} (flush) stored strategy:")
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    for i, (action, prob, regret) in enumerate(zip(actions, flush_strategy, flush_regrets)):
        print(f"  {action:12}: prob={prob:6.3f}, regret={regret:8.3f}")
    
    fold_prob = flush_strategy[0]
    allin_prob = flush_strategy[8]
    
    print(f"\n🚨 Strategy Check:")
    print(f"  FOLD prob: {fold_prob:.3f}")
    print(f"  ALL_IN prob: {allin_prob:.3f}")
    
    if fold_prob > allin_prob:
        print(f"  ❌ STORED STRATEGY IS BROKEN!")
        print(f"      Regrets are correct but strategy is wrong!")
    else:
        print(f"  ✅ Stored strategy prefers ALL_IN over FOLD")
        
    # Check if strategy is still uniform (untrained)
    uniform_check = abs(fold_prob - 1/9) < 0.05
    print(f"  Strategy uniform? {uniform_check} (fold_prob ≈ 0.111)")

except Exception as e:
    print(f"❌ Error loading model: {e}")


print("\n🔍 TESTING LEARNING RATE DECAY:")
print("=" * 40)

import math

learning_rate_base = 0.02
iterations = [1000, 10000, 20000, 50000]

for iter_num in iterations:
    lr = learning_rate_base / math.sqrt(iter_num + 1)
    print(f"Iteration {iter_num:5d}: LR = {lr:.8f}")

print(f"\n🚨 At 50K iterations:")
final_lr = learning_rate_base / math.sqrt(50000 + 1) 
print(f"   Learning rate: {final_lr:.8f}")
print(f"   Effective regret update: {final_lr * 1.222:.8f}")
print(f"   This is practically ZERO!")