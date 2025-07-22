from poker_bot import PokerBot

bot = PokerBot("models/learning_test.pkl")

# Test con manos específicas:
test_cases = [
    # Mano fuerte
    {"hole_cards": [51, 47], "pot_size": 50.0, "position": 2},  # AA
    # Mano débil  
    {"hole_cards": [0, 4], "pot_size": 50.0, "position": 2},   # 72o
    # Mano marginal
    {"hole_cards": [32, 33], "pot_size": 50.0, "position": 2}  # T9s
]

for i, test in enumerate(test_cases):
    action = bot.get_action(test)
    print(f"Test {i+1}: {action}") 