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