#!/usr/bin/env python3
"""
🔧 Debug Step-by-Step de _cfr_step_pure
Identifica exactamente dónde se produce el bloqueo paso a paso.
"""

import sys
import jax
import jax.numpy as jnp
import time

def debug_cfr_components():
    """Debuggear cada componente de _cfr_step_pure por separado"""
    print("🔧 DEBUG: Componentes de _cfr_step_pure")
    print("="*50)
    
    try:
        from poker_bot.core.trainer import TrainerConfig
        from poker_bot.core import full_game_engine as game_engine
        
        # Configuración minimal
        config = TrainerConfig(batch_size=1, max_info_sets=5, num_actions=3)
        
        # Arrays de prueba
        regrets = jnp.zeros((5, 3), dtype=jnp.float32)
        strategy = jnp.ones((5, 3), dtype=jnp.float32) / 3.0
        key = jax.random.PRNGKey(42)
        lut_keys = jnp.array([0, 1, 2], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300], dtype=jnp.int32)
        lut_table_size = 3
        
        print("✅ Setup completado")
        
        # PASO 1: Generar keys
        print("\n🔍 PASO 1: Generar random keys...")
        keys = jax.random.split(key, config.batch_size)
        print(f"✅ Keys generados: shape {keys.shape}")
        
        # PASO 2: Probar motor de juego DIRECTAMENTE
        print("\n🔍 PASO 2: Probar motor de juego...")
        start_time = time.time()
        
        try:
            payoffs, histories, game_results_batch = game_engine.unified_batch_simulation_with_lut(
                keys, lut_keys, lut_values, lut_table_size
            )
            elapsed = time.time() - start_time
            print(f"✅ Motor de juego completado en {elapsed:.2f}s")
            print(f"   Payoffs: {payoffs.shape}, Game results keys: {list(game_results_batch.keys())}")
            
        except Exception as e:
            print(f"❌ FALLO EN MOTOR DE JUEGO: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # PASO 3: Extraer datos del primer juego
        print("\n🔍 PASO 3: Extraer datos del juego...")
        try:
            game_payoffs = payoffs[0]  # [6] 
            game_results = {
                'hole_cards': game_results_batch['hole_cards'][0],  # [6, 2]
                'final_community': game_results_batch['final_community'][0],  # [5]
                'final_pot': game_results_batch['final_pot'][0],  # scalar
                'player_stacks': game_results_batch['player_stacks'][0],  # [6]
                'player_bets': game_results_batch['player_bets'][0]  # [6]
            }
            print(f"✅ Datos extraídos: payoffs shape {game_payoffs.shape}")
            print(f"   Hole cards: {game_results['hole_cards'].shape}")
            print(f"   Community: {game_results['final_community'].shape}")
            
        except Exception as e:
            print(f"❌ FALLO EXTRAYENDO DATOS: {e}")
            return False
        
        # PASO 4: Probar función de update de regrets
        print("\n🔍 PASO 4: Probar _update_regrets_for_game_pure...")
        try:
            from poker_bot.core.trainer import _update_regrets_for_game_pure
            
            start_time = time.time()
            regret_updates = _update_regrets_for_game_pure(
                regrets, game_results, game_payoffs, config.num_actions
            )
            elapsed = time.time() - start_time
            
            print(f"✅ Update regrets completado en {elapsed:.2f}s")
            print(f"   Regret updates shape: {regret_updates.shape}")
            print(f"   Max update: {jnp.max(jnp.abs(regret_updates)):.6f}")
            
        except Exception as e:
            print(f"❌ FALLO EN UPDATE REGRETS: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # PASO 5: Probar resto de operaciones CFR
        print("\n🔍 PASO 5: Probar operaciones CFR...")
        try:
            # Descuento de regrets
            discounted_regrets = jnp.where(
                config.use_regret_discounting,
                regrets * config.discount_factor,
                regrets
            )
            
            # Actualización
            updated_regrets = discounted_regrets + regret_updates
            
            # CFR+ pruning
            if config.use_cfr_plus:
                updated_regrets = jnp.maximum(updated_regrets, 0.0)
            
            # Regret matching
            from poker_bot.core.trainer import _regret_matching_pure
            updated_strategy = _regret_matching_pure(updated_regrets, config)
            
            print(f"✅ Operaciones CFR completadas")
            print(f"   Updated regrets: {updated_regrets.shape}")
            print(f"   Updated strategy: {updated_strategy.shape}")
            
        except Exception as e:
            print(f"❌ FALLO EN OPERACIONES CFR: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 ¡TODOS LOS COMPONENTES FUNCIONAN!")
        print("   El problema NO está en los componentes individuales.")
        print("   Debe ser un problema de compilación JIT de la función completa.")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jit_compilation_problem():
    """Verificar si el problema es específicamente la compilación JIT"""
    print("\n🔧 DEBUG: Problema de compilación JIT")
    print("="*50)
    
    try:
        print("1. Probando función SIN @jax.jit...")
        
        from poker_bot.core.trainer import TrainerConfig
        
        # Crear versión NO-JIT de _cfr_step_pure para testing
        def _cfr_step_no_jit(regrets, strategy, key, config, lut_keys, lut_values, lut_table_size):
            """Versión NO-JIT de _cfr_step_pure para debugging"""
            from poker_bot.core import full_game_engine as game_engine
            from poker_bot.core.trainer import _update_regrets_for_game_pure, _regret_matching_pure
            
            # Generar keys
            keys = jax.random.split(key, config.batch_size)
            
            # Motor de juego
            payoffs, histories, game_results_batch = game_engine.unified_batch_simulation_with_lut(
                keys, lut_keys, lut_values, lut_table_size
            )
            
            # Procesar primer juego
            game_payoffs = payoffs[0]
            game_results = {
                'hole_cards': game_results_batch['hole_cards'][0],
                'final_community': game_results_batch['final_community'][0],
                'final_pot': game_results_batch['final_pot'][0],
                'player_stacks': game_results_batch['player_stacks'][0],
                'player_bets': game_results_batch['player_bets'][0]
            }
            
            # Update regrets
            regret_updates = _update_regrets_for_game_pure(
                regrets, game_results, game_payoffs, config.num_actions
            )
            
            # Aplicar descuento y CFR+
            discounted_regrets = jnp.where(
                config.use_regret_discounting,
                regrets * config.discount_factor,
                regrets
            )
            
            updated_regrets = discounted_regrets + regret_updates
            
            if config.use_cfr_plus:
                updated_regrets = jnp.maximum(updated_regrets, 0.0)
            
            updated_strategy = _regret_matching_pure(updated_regrets, config)
            
            return updated_regrets, updated_strategy
        
        # Probar versión NO-JIT
        config = TrainerConfig(batch_size=1, max_info_sets=5, num_actions=3)
        regrets = jnp.zeros((5, 3), dtype=jnp.float32)
        strategy = jnp.ones((5, 3), dtype=jnp.float32) / 3.0
        key = jax.random.PRNGKey(42)
        lut_keys = jnp.array([0, 1, 2], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300], dtype=jnp.int32)
        
        print("   Ejecutando versión NO-JIT...")
        start_time = time.time()
        
        updated_regrets, updated_strategy = _cfr_step_no_jit(
            regrets, strategy, key, config, lut_keys, lut_values, 3
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Versión NO-JIT completada en {elapsed:.2f}s")
        
        # Verificar resultados
        regrets_changed = not jnp.allclose(regrets, updated_regrets, atol=1e-6)
        strategy_changed = not jnp.allclose(strategy, updated_strategy, atol=1e-6)
        
        print(f"   Regrets cambiaron: {regrets_changed}")
        print(f"   Strategy cambió: {strategy_changed}")
        
        return True
        
    except Exception as e:
        print(f"❌ FALLO EN VERSIÓN NO-JIT: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar debugging completo paso a paso"""
    print("🔧 CFR+ DEBUG STEP-BY-STEP")
    print("Objetivo: Encontrar EXACTAMENTE dónde se cuelga")
    print("\n")
    
    # Test 1: Componentes individuales
    success1 = debug_cfr_components()
    
    if success1:
        # Test 2: Problema de JIT
        success2 = test_jit_compilation_problem()
        
        if success2:
            print("\n🎯 DIAGNÓSTICO FINAL:")
            print("   ✅ Todos los componentes funcionan individualmente")
            print("   ✅ La función completa funciona SIN @jax.jit")
            print("   ❌ El problema está en la COMPILACIÓN JIT de @jax.jit")
            
            print("\n💡 SOLUCIÓN:")
            print("   1. Usar versión NO-JIT temporalmente para desarrollo")
            print("   2. Optimizar funciones JIT para compilación más rápida")
            print("   3. Usar compilación JIT progresiva")
            
            return True
        
    print("\n⚠️ PROBLEMAS DETECTADOS - revisar logs arriba")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)