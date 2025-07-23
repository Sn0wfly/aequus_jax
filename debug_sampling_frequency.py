# Debug para entender el problema de sparsity
import pickle
import jax.numpy as jnp
import numpy as np
from poker_bot.core.bucketing import compute_info_set_id

print("🔍 SPARSITY ANALYSIS:")
print("=" * 50)

# Cargar modelo entrenado
with open('models/convergence_test.pkl', 'rb') as f:
    model = pickle.load(f)

regrets = model['regrets']
strategy = model['strategy']

print(f"📊 Estadísticas del modelo:")
print(f"  Total info sets: {regrets.shape[0]:,}")
print(f"  Iterations: {model['iteration']:,}")

# Analizar cuántos info sets tienen regrets no-cero
non_zero_regrets = jnp.any(jnp.abs(regrets) > 0.001, axis=1)
num_visited = jnp.sum(non_zero_regrets)
coverage_percent = (num_visited / regrets.shape[0]) * 100

print(f"\n🎯 Cobertura de entrenamiento:")
print(f"  Info sets visitados: {num_visited:,}")
print(f"  Info sets totales: {regrets.shape[0]:,}")
print(f"  Cobertura: {coverage_percent:.2f}%")

# Analizar el info set problemático
target_info_set = 26015
target_regrets = regrets[target_info_set]
target_strategy = strategy[target_info_set]

print(f"\n🔍 Info Set {target_info_set} específico:")
print(f"  ¿Tiene regrets no-cero? {jnp.any(jnp.abs(target_regrets) > 0.001)}")
print(f"  Max regret abs: {jnp.max(jnp.abs(target_regrets)):.6f}")
print(f"  ¿Estrategia uniforme? {jnp.allclose(target_strategy, 1/9, atol=0.01)}")

# Encontrar info sets más visitados
regret_magnitudes = jnp.sum(jnp.abs(regrets), axis=1)
top_indices = jnp.argsort(regret_magnitudes)[-10:]  # Top 10

print(f"\n🏆 Top 10 info sets más entrenados:")
for i, idx in enumerate(top_indices[::-1]):
    magnitude = regret_magnitudes[idx]
    print(f"  {i+1:2}. Info set {idx:6}: magnitude {magnitude:8.3f}")

# Calcular visitas estimadas
batch_size = 128
total_iterations = model['iteration']
games_per_iteration = batch_size * 6  # 6 players per game
total_games = total_iterations * games_per_iteration

estimated_visits_per_infoset = total_games / regrets.shape[0]

print(f"\n📈 Estimación de visitas:")
print(f"  Games por iteración: {games_per_iteration:,}")
print(f"  Total games: {total_games:,}")
print(f"  Visitas promedio por info set: {estimated_visits_per_infoset:.1f}")

if estimated_visits_per_infoset < 10:
    print(f"  ⚠️  MUY POCAS VISITAS - Problema de sparsity!")
else:
    print(f"  ✅ Visitas suficientes")

# Verificar learning rate efectivo
learning_rate = 0.02  # Del config
print(f"\n🔧 Learning rate analysis:")
print(f"  Config learning rate: {learning_rate}")
print(f"  Action value FOLD: +2.311")
print(f"  Expected regret per visit: {2.311 * learning_rate:.3f}")
print(f"  Expected total regret (5 visits): {2.311 * learning_rate * 5:.3f}")
print(f"  Actual FOLD regret: {target_regrets[0]:.3f}")

if abs(target_regrets[0]) < 0.5:
    print(f"  ❌ Regrets mucho más pequeños de lo esperado!")
    print(f"  ➡️  Info set visitado muy pocas veces O bug en accumulation")
else:
    print(f"  ✅ Regrets en rango esperado")