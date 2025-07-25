#!/usr/bin/env python3
"""
Strategy Analyzer for Poker Bot
Analyzes exploitability of a pre-trained poker bot by testing its decisions
in two theoretically solved preflop scenarios.
"""

import jax.numpy as jnp
import numpy as np
import pickle
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.bucketing import compute_info_set_id

def load_bot_model(model_path):
    """Load the poker bot model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract strategy and config from the loaded model
        strategy = jnp.array(model_data['strategy'])
        config = model_data['config']
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Strategy shape: {strategy.shape}")
        print(f"   Max info sets: {config.max_info_sets}")
        
        return strategy, config
        
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def analyze_scenario(strategy, config, scenario_name, hole_cards, player_idx, 
                   community_cards, pot_size, max_info_sets):
    """Analyze a specific poker scenario and return strategy probabilities."""
    
    print(f"\n[ANÁLISIS DE ESTRATEGIA GTO]")
    print(f"ESCENARIO: {scenario_name}")
    
    # Calculate info set ID
    info_set_id = compute_info_set_id(
        hole_cards, community_cards, player_idx, pot_size, 
        max_info_sets=max_info_sets
    )
    
    print(f"Info Set ID: {info_set_id}")
    
    # Extract strategy probabilities for this info set
    if info_set_id < strategy.shape[0]:
        strategy_probs = strategy[info_set_id]
        strategy_probs = np.array(strategy_probs)  # Convert to numpy for easier handling
    else:
        print(f"⚠️  Warning: Info Set ID {info_set_id} exceeds strategy table size {strategy.shape[0]}")
        strategy_probs = np.ones(9) / 9  # Uniform strategy as fallback
    
    # Define action names
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    print("Estrategia Aprendida:")
    for i, (action, prob) in enumerate(zip(actions, strategy_probs)):
        print(f"{action}: {prob*100:.1f}%")
    
    return strategy_probs, actions

def evaluate_scenario_a(strategy_probs, actions):
    """Evaluate Scenario A (7-2o UTG) - should fold >95%."""
    fold_prob = strategy_probs[actions.index("FOLD")]
    
    print(f"Probabilidad de FOLD: {fold_prob*100:.1f}%")
    
    if fold_prob > 0.95:
        verdict = "¡ÉXITO! El bot foldea correctamente su mano basura (>95%)."
    else:
        verdict = "❌ FALLO: El bot no foldea suficientemente su mano basura."
    
    print(f"VEREDICTO: {verdict}")
    return fold_prob > 0.95

def evaluate_scenario_b(strategy_probs, actions):
    """Evaluate Scenario B (A-A BTN) - should play aggressively >95%."""
    aggressive_actions = ["BET_SMALL", "BET_MED", "BET_LARGE", "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    aggressive_prob = sum(strategy_probs[actions.index(action)] for action in aggressive_actions)
    
    print(f"Probabilidad de Acción Agresiva (Raise/Bet/All-in): {aggressive_prob*100:.1f}%")
    
    if aggressive_prob > 0.95:
        verdict = "¡ÉXITO! El bot juega agresivamente su mano monstruo (>95%)."
    else:
        verdict = "❌ FALLO: El bot no juega suficientemente agresivo con su mano monstruo."
    
    print(f"VEREDICTO: {verdict}")
    return aggressive_prob > 0.95

def main():
    """Main analysis function."""
    print("🎯 STRATEGY ANALYZER - Poker Bot GTO Analysis")
    print("=" * 60)
    
    # Load the bot model
    model_path = "models/pro_bot_v1.pkl"
    strategy, config = load_bot_model(model_path)
    
    if strategy is None or config is None:
        print("❌ Cannot proceed without valid model.")
        return
    
    # Common parameters for both scenarios
    community_cards = jnp.full(5, -1)  # Preflop
    pot_size = jnp.array([15.0])       # Blinds 5/10
    
    # Scenario A: 7-2o UTG (trash hand in early position)
    print("\n" + "=" * 60)
    hole_cards_a = jnp.array([23, 1])  # 7s, 2h
    player_idx_a = 3                   # UTG position
    strategy_probs_a, actions = analyze_scenario(
        strategy, config,
        "Mano Basura (7-2o) en Posición Temprana (UTG)",
        hole_cards_a, player_idx_a, community_cards, pot_size, config.max_info_sets
    )
    success_a = evaluate_scenario_a(strategy_probs_a, actions)
    
    # Scenario B: A-A BTN (monster hand in late position)
    print("\n" + "=" * 60)
    hole_cards_b = jnp.array([51, 49])  # Ac, Ah
    player_idx_b = 0                    # Button position
    strategy_probs_b, actions = analyze_scenario(
        strategy, config,
        "Mano Monstruo (A-A) en Posición Tardía (BTN)",
        hole_cards_b, player_idx_b, community_cards, pot_size, config.max_info_sets
    )
    success_b = evaluate_scenario_b(strategy_probs_b, actions)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL:")
    print(f"Escenario A (7-2o UTG): {'✅ ÉXITO' if success_a else '❌ FALLO'}")
    print(f"Escenario B (A-A BTN): {'✅ ÉXITO' if success_b else '❌ FALLO'}")
    
    if success_a and success_b:
        print("\n🎉 ¡EXCELENTE! El bot muestra comportamiento GTO correcto en ambos escenarios.")
    elif success_a or success_b:
        print("\n⚠️  MIXTO: El bot muestra comportamiento correcto en un escenario pero falla en el otro.")
    else:
        print("\n💀 CRÍTICO: El bot falla en ambos escenarios. Necesita más entrenamiento.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 