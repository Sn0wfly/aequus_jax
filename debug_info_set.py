# Debug script para verificar info sets y estrategias
import pickle
import jax.numpy as jnp
from poker_bot.core.bucketing import compute_info_set_id

# TESTING: verificar collision de info sets
print("🔍 TESTING INFO SET COLLISIONS:")

test_hands = [
    (jnp.array([51, 47]), "AA"),    # Pocket Aces
    (jnp.array([23, 0]), "72o"),    # Worst hand  
    (jnp.array([29, 6]), "Test"),   # Mano del problema
    (jnp.array([44, 45]), "KK"),    # Pocket Kings
]

comm_cards = jnp.array([34, 11, 3, 19, 41])
pot_size = jnp.array([50.0])

for hole, name in test_hands:
    info_id = compute_info_set_id(hole, comm_cards, 0, pot_size)
    print(f"  {name:4}: info_set = {info_id}")

print(f"⚠️  ¿Múltiples hands → mismo info set? ¡COLLISION!")
print()

# Cargar modelo entrenado
with open('models/final_conservative.pkl', 'rb') as f:
    model = pickle.load(f)

regrets = model['regrets']
strategy = model['strategy']

# Mano problemática del test
hole_cards = jnp.array([29, 6])  # strength = 0.067
comm_cards = jnp.array([34, 11, 3, 19, 41])
pot_size = jnp.array([50.0])
player_id = 0

# Calcular info set ID
info_set_id = compute_info_set_id(hole_cards, comm_cards, player_id, pot_size)
print(f"🔍 Debug Info Set Analysis:")
print(f"  Hole cards: {hole_cards}")
print(f"  Community: {comm_cards}")
print(f"  Info Set ID: {info_set_id}")

# Verificar regrets para este info set
if info_set_id < regrets.shape[0]:
    info_regrets = regrets[info_set_id]
    info_strategy = strategy[info_set_id]
    
    print(f"\n📊 Regrets para info set {info_set_id}:")
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    for i, (action, regret, prob) in enumerate(zip(actions, info_regrets, info_strategy)):
        print(f"  {action:12}: regret={regret:8.3f}, strategy={prob:8.3f}")
    
    print(f"\n🎯 Acción más probable: {actions[jnp.argmax(info_strategy)]}")
    print(f"  Probabilidad: {jnp.max(info_strategy):.3f}")
    
    # Verificar si los regrets están como esperamos
    fold_regret = info_regrets[0]
    bet_large_regret = info_regrets[5]
    
    print(f"\n🚨 Verificación de lógica:")
    print(f"  FOLD regret: {fold_regret:.3f} (debería ser ALTO)")
    print(f"  BET_LARGE regret: {bet_large_regret:.3f} (debería ser BAJO)")
    
    if fold_regret > bet_large_regret:
        print(f"  ✅ Regrets están correctos")
    else:
        print(f"  ❌ Regrets están INCORRECTOS!")
        
    # Strategy verification
    fold_prob = info_strategy[0]
    bet_large_prob = info_strategy[5]
    
    print(f"\n📈 Verificación de estrategia:")
    print(f"  FOLD prob: {fold_prob:.3f} (debería ser ALTA)")
    print(f"  BET_LARGE prob: {bet_large_prob:.3f} (debería ser BAJA)")
    
    if fold_prob > 0.5:
        print(f"  ✅ Estrategia prefiere FOLD")
    else:
        print(f"  ❌ Estrategia NO prefiere FOLD!")

else:
    print(f"❌ Info set ID {info_set_id} fuera de rango!")

# Estadísticas generales del modelo
total_regrets = jnp.sum(jnp.abs(regrets))
max_regret = jnp.max(regrets)
min_regret = jnp.min(regrets)

print(f"\n📊 Estadísticas del modelo:")
print(f"  Total regret magnitude: {total_regrets:.1f}")
print(f"  Max regret: {max_regret:.3f}")
print(f"  Min regret: {min_regret:.3f}")
print(f"  Iteration: {model['iteration']}")