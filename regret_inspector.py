#!/usr/bin/env python3
"""
Regret Inspector for Poker Bot
Inspects accumulated regret values for specific Info Sets to diagnose training coverage.
"""

import jax.numpy as jnp
import numpy as np
import pickle
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_bot_model(model_path):
    """Load the poker bot model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract regrets and config from the loaded model
        regrets = jnp.array(model_data['regrets'])
        config = model_data['config']
        
        print(f"✅ Modelo '{os.path.basename(model_path)}' cargado.")
        print(f"   Regrets shape: {regrets.shape}")
        print(f"   Max info sets: {config.max_info_sets}")
        
        return regrets, config
        
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def inspect_info_set(regrets, info_set_id, scenario_name):
    """Inspect a specific Info Set and analyze its regret values."""
    
    print(f"\nESCENARIO: {scenario_name}")
    print(f"Info Set ID: {info_set_id}")
    
    # Check if info_set_id is within bounds
    if info_set_id >= regrets.shape[0]:
        print(f"⚠️  Warning: Info Set ID {info_set_id} exceeds regrets table size {regrets.shape[0]}")
        return None
    
    # Extract regret vector for this info set
    regret_vector = regrets[info_set_id]
    regret_vector_np = np.array(regret_vector)  # Convert to numpy for easier handling
    
    # Display the regret vector
    print(f"Vector de Regrets: {regret_vector_np}")
    
    # Calculate total regret magnitude (sum of absolute values)
    regret_magnitude = np.sum(np.abs(regret_vector_np))
    print(f"Magnitud Total de Regret: {regret_magnitude:.2f}")
    
    # Determine verdict based on regret magnitude
    if regret_magnitude == 0.0:
        verdict = "❌ NO ENTRENADO. Los regrets son cero, explicando la estrategia uniforme."
        was_trained = False
    else:
        verdict = "✅ ENTRENADO. Los regrets no son cero. El fallo está en otro lado."
        was_trained = True
    
    print(f"VEREDICTO: {verdict}")
    
    return regret_vector_np, regret_magnitude, was_trained

def provide_diagnosis(scenario_a_result, scenario_b_result):
    """Provide final diagnosis based on inspection results."""
    
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO FINAL:")
    
    if scenario_a_result is None or scenario_b_result is None:
        print("❌ No se pudo completar el análisis debido a errores en la carga del modelo.")
        return
    
    _, magnitude_a, trained_a = scenario_a_result
    _, magnitude_b, trained_b = scenario_b_result
    
    if not trained_a and not trained_b:
        print("💀 CRÍTICO: Ninguno de los Info Sets fue entrenado.")
        print("   - El modelo no ha visto estas situaciones durante el entrenamiento.")
        print("   - Necesitas más iteraciones de entrenamiento o ajustar el bucketing.")
        print("   - Considera aumentar el batch_size o el número de iteraciones.")
        
    elif not trained_a and trained_b:
        print("⚠️  MIXTO: Solo el Info Set B (A-A BTN) fue entrenado.")
        print("   - El Info Set A (7-2o UTG) no fue encontrado durante el entrenamiento.")
        print("   - Esto sugiere que las manos basura no se están generando suficientemente.")
        print("   - Considera ajustar la distribución de manos en el game engine.")
        
    elif trained_a and not trained_b:
        print("⚠️  MIXTO: Solo el Info Set A (7-2o UTG) fue entrenado.")
        print("   - El Info Set B (A-A BTN) no fue encontrado durante el entrenamiento.")
        print("   - Esto sugiere que las manos premium no se están generando suficientemente.")
        print("   - Considera ajustar la distribución de manos en el game engine.")
        
    else:  # Both trained
        print("✅ AMBOS ENTRENADOS: Los Info Sets fueron encontrados durante el entrenamiento.")
        print("   - El problema no es de cobertura de entrenamiento.")
        print("   - El fallo en strategy_analyzer.py puede deberse a:")
        print("     * Configuración incorrecta de CFR+ o regret discounting")
        print("     * Problemas en el cálculo de estrategia")
        print("     * Valores de learning_rate inadecuados")
        print("     * Errores en el bucketing system")
    
    # Additional insights
    print(f"\n📊 INSIGHTS ADICIONALES:")
    print(f"   Magnitud A (7-2o UTG): {magnitude_a:.2f}")
    print(f"   Magnitud B (A-A BTN): {magnitude_b:.2f}")
    
    if magnitude_a > 0 and magnitude_b > 0:
        ratio = magnitude_b / magnitude_a if magnitude_a > 0 else float('inf')
        print(f"   Ratio B/A: {ratio:.2f}")
        if ratio > 10:
            print("   ✅ El ratio es alto, lo cual es esperado (A-A debería tener regrets más altos).")
        else:
            print("   ⚠️  El ratio es bajo, podría indicar problemas en el entrenamiento.")

def main():
    """Main inspection function."""
    print("🔍 REGRET INSPECTOR - Análisis de Cobertura de Entrenamiento")
    print("=" * 60)
    
    # Load the bot model - try both possible filenames
    model_paths = [
        "models/pro_bot_final_cfr_plus.pkl",
        "models/pro_bot_v1.pkl"
    ]
    
    regrets = None
    config = None
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            regrets, config = load_bot_model(model_path)
            if regrets is not None:
                break
    
    if regrets is None or config is None:
        print("❌ No se pudo cargar ningún modelo. Verifica que exista uno de estos archivos:")
        for path in model_paths:
            print(f"   - {path}")
        return
    
    # Define the Info Sets to inspect (from strategy_analyzer.py failures)
    info_sets_to_inspect = [
        (2903, "Info Set 2903 (7-2o UTG)"),
        (24003, "Info Set 24003 (A-A BTN)")
    ]
    
    # Inspect each Info Set
    results = []
    for info_set_id, scenario_name in info_sets_to_inspect:
        result = inspect_info_set(regrets, info_set_id, scenario_name)
        results.append(result)
        print("\n" + "=" * 60)
    
    # Provide final diagnosis
    if len(results) == 2:
        provide_diagnosis(results[0], results[1])
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 