# Debug del training loop COMPLETO incluyendo _cfr_step_pure
import jax
import jax.numpy as jnp
from poker_bot.core.trainer import _cfr_step_pure, TrainerConfig

print("🔍 FULL TRAINING LOOP DEBUG:")
print("=" * 50)

# Setup inicial idéntico al trainer real
config = TrainerConfig(
    batch_size=128,
    num_actions=9,
    max_info_sets=50000,
    learning_rate=0.02,
    use_cfr_plus=False,
    use_regret_discounting=False
)

print(f"📊 Config:")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Batch size: {config.batch_size}")

# Estado inicial
regrets = jnp.zeros((50000, 9))
strategy = jnp.ones((50000, 9)) / 9

print(f"\n🎯 Info Set 26027 estado inicial:")
initial_regrets = regrets[26027]
print(f"  Initial regrets: {jnp.max(jnp.abs(initial_regrets)):.6f} (debería ser 0)")

# Ejecutar UNA iteración completa de _cfr_step_pure
print(f"\n🚀 Ejecutando _cfr_step_pure completo...")
key = jax.random.PRNGKey(42)

# CRITICAL: Llamar la función REAL de training
updated_regrets, updated_strategy = _cfr_step_pure(regrets, strategy, key, config)

print(f"\n📈 Info Set 26027 después de 1 iteración:")
final_regrets = updated_regrets[26027]
regret_changes = final_regrets - initial_regrets

actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]

print(f"📊 Cambios en regrets:")
for action, change in zip(actions, regret_changes):
    print(f"  {action:12}: {change:8.6f}")

# Verificar magnitud esperada
fold_change = regret_changes[0]
expected_change = 2.311 * 0.02  # action_value × learning_rate

print(f"\n🚨 Verificación final:")
print(f"  Expected FOLD change: ~{expected_change:.6f}")
print(f"  Actual FOLD change: {fold_change:.6f}")
print(f"  Ratio: {fold_change/expected_change if expected_change != 0 else 'N/A':.3f}")

if abs(fold_change - expected_change) < 0.01:
    print(f"  ✅ Learning rate se aplica correctamente!")
    print(f"  ➡️  El problema debe ser SPARSITY (pocas visitas)")
else:
    print(f"  ❌ Learning rate aún no se aplica!")
    print(f"  ➡️  Hay un bug más profundo en el training loop")

# Additional analysis: cuántos info sets fueron afectados
non_zero_changes = jnp.sum(jnp.any(jnp.abs(updated_regrets - regrets) > 0.001, axis=1))
print(f"\n📊 Estadísticas de la iteración:")
print(f"  Info sets modificados: {non_zero_changes}")
print(f"  Info sets totales: {regrets.shape[0]}")
print(f"  Cobertura por iteración: {non_zero_changes/regrets.shape[0]*100:.2f}%")

# Verificar si nuestro info set específico fue tocado
info_set_touched = jnp.any(jnp.abs(regret_changes) > 0.001)
print(f"  Info set 26027 fue modificado: {info_set_touched}")

if info_set_touched:
    print(f"  ✅ El info set SÍ se entrenó en esta iteración")
else:
    print(f"  ❌ El info set NO se entrenó (SPARSITY confirmado)")