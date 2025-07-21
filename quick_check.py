from poker_bot import create_trainer

trainer = create_trainer()
trainer.load_model("models/learning_test.pkl")

# Check regrets
import jax.numpy as jnp
from poker_bot.core.bucketing import compute_info_set_id

# Test qué info set IDs genera el bucketing
test_hole_cards = jnp.array([48, 49])  # AA
test_community = jnp.full(5, -1)
test_pot = jnp.array([50.0])

info_set_id = compute_info_set_id(test_hole_cards, test_community, 0, test_pot)
print(f"Example info set ID: {info_set_id}")
print(f"Max possible ID range: 0-{trainer.regrets.shape[0]-1}")

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

print("\n🔍 DEBUGGING GAME DISTRIBUTION:")

# Encuentra los info sets que tienen regrets positivos
def get_positive_info_sets(regrets):
    positive_info_sets = []
    for i in range(regrets.shape[0]):
        if jnp.any(regrets[i] > 0):
            positive_info_sets.append(int(i))
    return positive_info_sets

positive_info_sets = get_positive_info_sets(trainer.regrets)
print(f"Info sets with positive regrets: {positive_info_sets[:10]}...")  # Primeros 10
if positive_info_sets:
    print(f"Range of positive info sets: {min(positive_info_sets)} - {max(positive_info_sets)}")
    print(f"Are they clustered? Gap analysis:")
    if len(positive_info_sets) > 1:
        gaps = [positive_info_sets[i+1] - positive_info_sets[i] for i in range(len(positive_info_sets)-1)]
        print(f"Avg gap between positive info sets: {sum(gaps)/len(gaps):.0f}")
        print(f"Min/Max gaps: {min(gaps)}/{max(gaps)}")

print(f"\n🔍 ANALYZING REGION 438K-443K:")

# Test diferentes situaciones para ver sus info set IDs
test_cases = [
    (jnp.array([51, 47]), "AA", 0),     # AA, position 0
    (jnp.array([51, 47]), "AA", 5),     # AA, position 5
    (jnp.array([23, 0]), "72o", 0),     # 72o, position 0
    (jnp.array([44, 45]), "KK", 2),     # KK, position 2
]

for hole_cards, name, pos in test_cases:
    test_community = jnp.full(5, -1)
    test_pot = jnp.array([50.0])
    info_id = compute_info_set_id(hole_cards, test_community, pos, test_pot)
    print(f"{name} pos {pos}: info set {info_id}")