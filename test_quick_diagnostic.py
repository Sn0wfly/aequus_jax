#!/usr/bin/env python3
"""
🔍 Diagnóstico Rápido - Problemas Específicos
Script para investigar los problemas identificados en el diagnóstico completo.
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_trainer_config_fix():
    """Verificar el fix del TrainerConfig"""
    print("🔧 TEST: TrainerConfig (corregido)")
    
    try:
        from poker_bot.core.trainer import TrainerConfig
        
        # Crear config SIN num_iterations
        config = TrainerConfig(
            batch_size=2,
            max_info_sets=100,
            save_interval=1000
        )
        print(f"✅ TrainerConfig creado: batch_size={config.batch_size}")
        return True, config
        
    except Exception as e:
        print(f"❌ Error en TrainerConfig: {e}")
        return False, None

def test_trainer_instantiation():
    """Verificar instanciación del trainer con LUT real"""
    print("\n🏗️ TEST: PokerTrainer instantiation")
    
    try:
        from poker_bot.core.trainer import PokerTrainer, TrainerConfig
        
        config = TrainerConfig(batch_size=1, max_info_sets=50)
        trainer = PokerTrainer(config)
        
        print(f"✅ Trainer creado con {trainer.lut_table_size:,} LUT entries")
        print(f"   Regrets shape: {trainer.regrets.shape}")
        print(f"   Strategy shape: {trainer.strategy.shape}")
        return True, trainer
        
    except Exception as e:
        print(f"❌ Error instanciando trainer: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_game_engine_basic():
    """Probar motor de juego sin JIT para ver si hay problemas básicos"""
    print("\n🎮 TEST: Motor de juego básico (sin JIT)")
    
    try:
        from poker_bot.core import full_game_engine as game_engine
        
        # Test muy básico: evaluar una mano simple
        keys = jax.random.split(jax.random.PRNGKey(42), 1)
        lut_keys = jnp.array([0, 1], dtype=jnp.int32)
        lut_values = jnp.array([100, 200], dtype=jnp.int32)
        
        print("   Preparando datos de prueba...")
        
        # Solo verificar que la función existe y es callable
        if hasattr(game_engine, 'unified_batch_simulation_with_lut'):
            print("✅ unified_batch_simulation_with_lut disponible")
        else:
            print("❌ unified_batch_simulation_with_lut NO disponible")
            return False
            
        # NO ejecutar la función aún - solo verificar preparación
        print("✅ Motor de juego preparado para test")
        return True
        
    except Exception as e:
        print(f"❌ Error en motor de juego básico: {e}")
        return False

def test_minimal_cfr_data():
    """Preparar datos mínimos para CFR sin ejecutar la función pura"""
    print("\n📊 TEST: Preparación datos CFR")
    
    try:
        from poker_bot.core.trainer import TrainerConfig
        
        config = TrainerConfig(batch_size=1, max_info_sets=10, num_actions=3)  # Muy pequeño
        
        # Arrays mínimos
        regrets = jnp.zeros((10, 3), dtype=jnp.float32)
        strategy = jnp.ones((10, 3), dtype=jnp.float32) / 3.0
        key = jax.random.PRNGKey(42)
        lut_keys = jnp.array([0, 1, 2], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300], dtype=jnp.int32)
        
        print(f"✅ Datos CFR preparados:")
        print(f"   Config: batch_size={config.batch_size}, info_sets={config.max_info_sets}")
        print(f"   Regrets: {regrets.shape}, Strategy: {strategy.shape}")
        print(f"   LUT: {len(lut_keys)} entries")
        
        return True, (regrets, strategy, key, config, lut_keys, lut_values, 3)
        
    except Exception as e:
        print(f"❌ Error preparando datos CFR: {e}")
        return False, None

def test_individual_functions():
    """Probar funciones individuales antes de la función pura completa"""
    print("\n🧩 TEST: Funciones individuales")
    
    try:
        from poker_bot.core.trainer import _evaluate_hand_simple_pure, _regret_matching_pure, TrainerConfig
        
        # Test 1: Evaluación de mano simple
        hole_cards = jnp.array([48, 44], dtype=jnp.int32)  # As, Ks
        strength = _evaluate_hand_simple_pure(hole_cards)
        print(f"✅ _evaluate_hand_simple_pure: {strength}")
        
        # Test 2: Regret matching
        config = TrainerConfig()
        test_regrets = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=jnp.float32)
        strategy = _regret_matching_pure(test_regrets, config)
        print(f"✅ _regret_matching_pure: shape {strategy.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en funciones individuales: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar diagnóstico rápido y específico"""
    print("🔍 DIAGNÓSTICO RÁPIDO - Problemas Específicos")
    print("="*50)
    
    tests = [
        ("TrainerConfig Fix", test_trainer_config_fix),
        ("Trainer Instantiation", test_trainer_instantiation), 
        ("Game Engine Basic", test_game_engine_basic),
        ("CFR Data Preparation", test_minimal_cfr_data),
        ("Individual Functions", test_individual_functions)
    ]
    
    results = {}
    data_cache = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                success, data = result
                if data is not None:
                    data_cache[test_name] = data
            else:
                success = result
            
            results[test_name] = success
            
        except Exception as e:
            print(f"❌ {test_name} EXCEPTION: {e}")
            results[test_name] = False
    
    # Resumen
    print("\n" + "="*50)
    print("📊 RESUMEN DIAGNÓSTICO RÁPIDO")
    print("="*50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResultado: {passed}/{total} tests pasaron")
    
    if passed >= 4:  # La mayoría pasó
        print("\n🎯 DIAGNÓSTICO: Componentes básicos funcionan.")
        print("   El problema parece estar en la ejecución de _cfr_step_pure.")
        print("   Posible causa: Motor de juego demasiado pesado para tests simples.")
        
        print("\n💡 SOLUCIONES RECOMENDADAS:")
        print("   1. Usar mock/dummy data para tests iniciales")
        print("   2. Reducir drasticamente batch_size y max_info_sets") 
        print("   3. Agregar timeouts a las funciones JIT")
        print("   4. Verificar que la GPU tenga suficiente memoria")
        
    else:
        print(f"\n⚠️ DIAGNÓSTICO: {total-passed} componentes básicos fallan.")
        print("   Revisar errores arriba antes de proceder.")
    
    return passed >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)