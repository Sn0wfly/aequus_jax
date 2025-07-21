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

print(f"Total info sets: {trainer.regrets.shape[0]}")
print(f"Positive regret coverage: {positive_regrets/trainer.regrets.shape[0]*100:.3f}%")

# Check estrategia en info sets con regrets positivos
positive_mask = trainer.regrets > 0
if jnp.any(positive_mask):
    positive_strategies = trainer.strategy[positive_mask]
    print(f"Strategy entropy in positive regret areas: {jnp.mean(-jnp.sum(positive_strategies * jnp.log(positive_strategies + 1e-10), axis=1)):.4f}") 