from poker_bot import create_trainer

trainer = create_trainer()
trainer.load_model("models/learning_test.pkl")

# Check regrets
import jax.numpy as jnp
positive_regrets = jnp.sum(trainer.regrets > 0)
max_regret = jnp.max(trainer.regrets)
min_regret = jnp.min(trainer.regrets)

print(f"Positive regrets: {positive_regrets}")
print(f"Max regret: {max_regret}")
print(f"Min regret: {min_regret}")
print(f"Strategy entropy: {trainer._compute_strategy_entropy()}")

# NUEVO ANÁLISIS CORREGIDO:
print(f"Total info sets: {trainer.regrets.shape[0]}")
print(f"Positive regret coverage: {positive_regrets/trainer.regrets.shape[0]*100:.3f}%")

# Check info sets que tienen AL MENOS un regret positivo
info_sets_with_positive = jnp.any(trainer.regrets > 0, axis=1)
num_info_sets_with_positive = jnp.sum(info_sets_with_positive)
print(f"Info sets with ANY positive regret: {num_info_sets_with_positive}")

# Strategy entropy en esos info sets
if num_info_sets_with_positive > 0:
    positive_info_set_strategies = trainer.strategy[info_sets_with_positive]
    entropies = -jnp.sum(positive_info_set_strategies * jnp.log(positive_info_set_strategies + 1e-10), axis=1)
    avg_entropy = jnp.mean(entropies)
    print(f"Avg strategy entropy in positive regret info sets: {avg_entropy:.4f}")
    print(f"Min/Max entropy in positive areas: {jnp.min(entropies):.4f}/{jnp.max(entropies):.4f}")

print(f"Trainer config MC sampling rate: {trainer.config.mc_sampling_rate}")
print(f"Current regret shape: {trainer.regrets.shape}")
print(f"Model iteration: {trainer.iteration}")
print(f"Trainer batch size: {trainer.config.batch_size}")