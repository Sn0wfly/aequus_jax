#!/usr/bin/env python3
"""
🔍 Diagnóstico de Integración Final - Poker AI
Script para verificar la integración del motor de juego real con el trainer.

Verificaciones:
1. Importaciones del motor de juego
2. Carga de LUT (load_hand_evaluation_lut)  
3. Instanciación del trainer
4. Función pura _cfr_step_pure
5. Integración con unified_batch_simulation_with_lut
"""

import sys
import traceback
import logging
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator(title: str):
    """Imprime un separador visual para las secciones del diagnóstico"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_result(test_name: str, success: bool, details: str = ""):
    """Imprime el resultado de una prueba"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"   Details: {details}")

def test_basic_imports():
    """Test 1: Verificar importaciones básicas"""
    print_separator("TEST 1: Importaciones Básicas")
    
    try:
        import poker_bot
        print_result("poker_bot import", True, "Módulo principal disponible")
    except Exception as e:
        print_result("poker_bot import", False, str(e))
        return False
    
    try:
        import poker_bot.core
        print_result("poker_bot.core import", True, "Módulo core disponible")
    except Exception as e:
        print_result("poker_bot.core import", False, str(e))
        return False
    
    try:
        from poker_bot.core import trainer
        print_result("trainer module import", True, "Módulo trainer disponible")
    except Exception as e:
        print_result("trainer module import", False, str(e))
        return False
        
    return True

def test_game_engine_imports():
    """Test 2: Verificar importaciones del motor de juego"""
    print_separator("TEST 2: Importaciones del Motor de Juego")
    
    try:
        from poker_bot.core import full_game_engine as game_engine
        print_result("full_game_engine import", True, "Motor de juego importado")
    except Exception as e:
        print_result("full_game_engine import", False, str(e))
        return False, None
    
    # Verificar funciones específicas necesarias
    required_functions = [
        'unified_batch_simulation_with_lut',
        'evaluate_hand_jax_native',
        'play_one_game',
        'batch_play'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if hasattr(game_engine, func_name):
            print_result(f"Function {func_name}", True, "Disponible en game_engine")
        else:
            print_result(f"Function {func_name}", False, "No encontrada en game_engine")
            missing_functions.append(func_name)
    
    success = len(missing_functions) == 0
    return success, game_engine

def test_lut_loading():
    """Test 3: Verificar carga de LUT"""
    print_separator("TEST 3: Carga de LUT (Load Hand Evaluation LUT)")
    
    try:
        from poker_bot.core.trainer import load_hand_evaluation_lut
        print_result("load_hand_evaluation_lut import", True, "Función disponible")
    except Exception as e:
        print_result("load_hand_evaluation_lut import", False, str(e))
        return False, None, None, None
    
    # Intentar cargar LUT
    try:
        lut_keys, lut_values, lut_table_size = load_hand_evaluation_lut()
        print_result("LUT loading", True, 
                    f"Cargado: {len(lut_keys)} keys, table_size={lut_table_size}")
        
        # Verificar tipos
        if isinstance(lut_keys, np.ndarray) and isinstance(lut_values, np.ndarray):
            print_result("LUT data types", True, 
                        f"Keys: {lut_keys.dtype}, Values: {lut_values.dtype}")
        else:
            print_result("LUT data types", False, 
                        f"Keys: {type(lut_keys)}, Values: {type(lut_values)}")
        
        return True, lut_keys, lut_values, lut_table_size
        
    except Exception as e:
        print_result("LUT loading", False, f"Error: {str(e)}")
        logger.error(f"LUT loading failed: {traceback.format_exc()}")
        return False, None, None, None

def test_trainer_instantiation():
    """Test 4: Verificar instanciación del trainer"""
    print_separator("TEST 4: Instanciación del Trainer")
    
    try:
        from poker_bot.core.trainer import PokerTrainer, TrainerConfig
        print_result("PokerTrainer classes import", True, "Clases disponibles")
    except Exception as e:
        print_result("PokerTrainer classes import", False, str(e))
        return False, None
    
    try:
        # Crear configuración básica
        config = TrainerConfig(
            batch_size=2,  # Muy pequeño para tests
            num_iterations=1,
            max_info_sets=1000,
            save_interval=1000
        )
        print_result("TrainerConfig creation", True, f"batch_size={config.batch_size}")
    except Exception as e:
        print_result("TrainerConfig creation", False, str(e))
        return False, None
    
    try:
        # Instanciar trainer - esto debería cargar la LUT automáticamente
        trainer = PokerTrainer(config)
        print_result("PokerTrainer instantiation", True, 
                    f"Trainer creado con {trainer.lut_table_size} LUT entries")
        return True, trainer
    except Exception as e:
        print_result("PokerTrainer instantiation", False, f"Error: {str(e)}")
        logger.error(f"Trainer instantiation failed: {traceback.format_exc()}")
        return False, None

def test_pure_function():
    """Test 5: Verificar función pura _cfr_step_pure"""
    print_separator("TEST 5: Función Pura _cfr_step_pure")
    
    try:
        from poker_bot.core.trainer import _cfr_step_pure, TrainerConfig
        print_result("_cfr_step_pure import", True, "Función pura disponible")
    except Exception as e:
        print_result("_cfr_step_pure import", False, str(e))
        return False
    
    try:
        # Configurar parámetros mínimos para la prueba
        config = TrainerConfig(
            batch_size=1,  # Solo 1 juego para test
            num_actions=6,
            max_info_sets=100
        )
        
        # Crear arrays de prueba
        regrets = jnp.zeros((100, 6), dtype=jnp.float32)
        strategy = jnp.ones((100, 6), dtype=jnp.float32) / 6.0
        key = jax.random.PRNGKey(42)
        
        # LUT de prueba (dummy data)
        lut_keys = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300, 400, 500, 600], dtype=jnp.int32)
        lut_table_size = 6
        
        print_result("Test data setup", True, "Arrays de prueba creados")
        
    except Exception as e:
        print_result("Test data setup", False, f"Error: {str(e)}")
        return False
    
    try:
        # Llamar a la función pura - esto debería usar el motor de juego real
        updated_regrets, updated_strategy = _cfr_step_pure(
            regrets, 
            strategy, 
            key, 
            config,
            lut_keys,
            lut_values,
            lut_table_size
        )
        
        print_result("_cfr_step_pure execution", True, 
                    f"Shapes: regrets{updated_regrets.shape}, strategy{updated_strategy.shape}")
        
        # Verificar que los arrays fueron actualizados
        regrets_changed = not jnp.allclose(regrets, updated_regrets)
        strategy_changed = not jnp.allclose(strategy, updated_strategy)
        
        print_result("Regrets updated", regrets_changed, 
                    f"Max regret: {jnp.max(jnp.abs(updated_regrets)):.6f}")
        print_result("Strategy updated", strategy_changed,
                    f"Strategy entropy: {-jnp.mean(jnp.sum(updated_strategy * jnp.log(updated_strategy + 1e-10), axis=1)):.6f}")
        
        return True
        
    except Exception as e:
        print_result("_cfr_step_pure execution", False, f"Error: {str(e)}")
        logger.error(f"_cfr_step_pure failed: {traceback.format_exc()}")
        return False

def test_game_engine_integration():
    """Test 6: Verificar integración con unified_batch_simulation_with_lut"""
    print_separator("TEST 6: Integración Motor de Juego")
    
    try:
        from poker_bot.core import full_game_engine as game_engine
        
        # Preparar datos de prueba
        keys = jax.random.split(jax.random.PRNGKey(42), 2)  # 2 juegos
        lut_keys = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300, 400, 500, 600], dtype=jnp.int32)
        lut_table_size = 6
        
        print_result("Game engine test setup", True, "Datos de prueba preparados")
        
    except Exception as e:
        print_result("Game engine test setup", False, str(e))
        return False
    
    try:
        # Llamar al motor de juego unificado
        payoffs, histories, game_results = game_engine.unified_batch_simulation_with_lut(
            keys, lut_keys, lut_values, lut_table_size
        )
        
        print_result("unified_batch_simulation_with_lut", True,
                    f"Payoffs shape: {payoffs.shape}, Histories: {histories.shape}")
        
        # Verificar estructura de game_results
        expected_keys = ['hole_cards', 'final_community', 'final_pot', 'player_stacks', 'player_bets']
        missing_keys = [k for k in expected_keys if k not in game_results]
        
        if not missing_keys:
            print_result("Game results structure", True, 
                        f"Todas las claves presentes: {list(game_results.keys())}")
        else:
            print_result("Game results structure", False, 
                        f"Claves faltantes: {missing_keys}")
        
        # Verificar dimensiones
        batch_size = keys.shape[0]
        hole_cards_shape = game_results['hole_cards'].shape
        expected_hole_shape = (batch_size, 6, 2)
        
        if hole_cards_shape == expected_hole_shape:
            print_result("Hole cards shape", True, f"Shape correcto: {hole_cards_shape}")
        else:
            print_result("Hole cards shape", False, 
                        f"Esperado: {expected_hole_shape}, Obtenido: {hole_cards_shape}")
        
        return True
        
    except Exception as e:
        print_result("unified_batch_simulation_with_lut", False, f"Error: {str(e)}")
        logger.error(f"Game engine integration failed: {traceback.format_exc()}")
        return False

def test_end_to_end():
    """Test 7: Prueba end-to-end del entrenamiento"""
    print_separator("TEST 7: Prueba End-to-End")
    
    try:
        from poker_bot.core.trainer import PokerTrainer, TrainerConfig
        
        # Configuración mínima para prueba rápida
        config = TrainerConfig(
            batch_size=1,
            max_info_sets=50,
            num_actions=6,
            save_interval=10,
            log_interval=1
        )
        
        # Crear trainer
        trainer = PokerTrainer(config)
        print_result("End-to-end trainer setup", True, "Trainer creado para prueba")
        
        # Simular 1 iteración de entrenamiento
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")
            
            # NOTA: Solo 1 iteración para verificar que todo funciona
            stats = trainer.train(num_iterations=1, save_path=save_path)
            
            print_result("Single training iteration", True, 
                        f"Completada en {stats['total_time']:.2f}s")
            
            # Verificar que se generaron estadísticas
            expected_stat_keys = ['iterations_completed', 'total_time', 'final_regret_sum']
            missing_stat_keys = [k for k in expected_stat_keys if k not in stats]
            
            if not missing_stat_keys:
                print_result("Training statistics", True, 
                            f"Stats generadas: {list(stats.keys())}")
            else:
                print_result("Training statistics", False, 
                            f"Stats faltantes: {missing_stat_keys}")
        
        return True
        
    except Exception as e:
        print_result("End-to-end test", False, f"Error: {str(e)}")
        logger.error(f"End-to-end test failed: {traceback.format_exc()}")
        return False

def main():
    """Ejecutar diagnóstico completo"""
    print_separator("DIAGNÓSTICO DE INTEGRACIÓN FINAL")
    print("Verificando integración del motor de juego real con el trainer CFR+")
    
    # Información del sistema
    print(f"\nSistema JAX: {jax.__version__}")
    print(f"Dispositivos disponibles: {jax.devices()}")
    print(f"Plataforma: {jax.default_backend()}")
    
    # Ejecutar todos los tests
    tests = [
        ("Importaciones Básicas", test_basic_imports),
        ("Motor de Juego", test_game_engine_imports),
        ("Carga de LUT", test_lut_loading),
        ("Instanciación Trainer", test_trainer_instantiation),
        ("Función Pura", test_pure_function),
        ("Integración Motor", test_game_engine_integration),
        ("End-to-End", test_end_to_end)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            # Manejar diferentes tipos de retorno
            if isinstance(result, tuple):
                success = result[0]
            else:
                success = result
            results[test_name] = success
        except Exception as e:
            print_result(f"{test_name} (EXCEPTION)", False, str(e))
            results[test_name] = False
    
    # Resumen final
    print_separator("RESUMEN DEL DIAGNÓSTICO")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Resultados: {passed}/{total} tests pasaron")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 ¡DIAGNÓSTICO COMPLETO! Todos los componentes funcionan correctamente.")
        print("   La integración del motor de juego real está lista para entrenamiento.")
    else:
        print(f"\n⚠️ DIAGNÓSTICO PARCIAL: {total-passed} componentes requieren atención.")
        print("   Revisa los errores arriba para solucionar los problemas.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)