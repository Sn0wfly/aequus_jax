#!/usr/bin/env python3
"""
🎯 Test Aislado de _cfr_step_pure
Prueba la función pura con configuración ultra-ligera para detectar el bloqueo.
"""

import sys
import jax
import jax.numpy as jnp
import time
import signal
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    """Context manager para timeout de funciones"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operación excedió {duration} segundos")
    
    # Configurar el handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        signal.alarm(0)

def test_cfr_step_minimal():
    """Probar _cfr_step_pure con la configuración más pequeña posible"""
    print("🎯 TEST: _cfr_step_pure con configuración ultra-minimal")
    print("="*60)
    
    try:
        from poker_bot.core.trainer import _cfr_step_pure, TrainerConfig
        
        # Configuración ULTRA-MINIMAL
        config = TrainerConfig(
            batch_size=1,           # Solo 1 juego
            max_info_sets=5,        # Solo 5 info sets  
            num_actions=3,          # Solo 3 acciones (fold, call, bet)
        )
        
        print(f"✅ Config: batch={config.batch_size}, info_sets={config.max_info_sets}, actions={config.num_actions}")
        
        # Arrays mínimos
        regrets = jnp.zeros((5, 3), dtype=jnp.float32)
        strategy = jnp.ones((5, 3), dtype=jnp.float32) / 3.0
        key = jax.random.PRNGKey(42)
        
        # LUT ultra-simple
        lut_keys = jnp.array([0, 1, 2], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300], dtype=jnp.int32)
        lut_table_size = 3
        
        print(f"✅ Arrays preparados: regrets{regrets.shape}, strategy{strategy.shape}")
        print(f"✅ LUT simple: {len(lut_keys)} entries")
        
        # PRUEBA CON TIMEOUT DE 30 SEGUNDOS
        print("\n⏰ Iniciando _cfr_step_pure con timeout de 30s...")
        start_time = time.time()
        
        with timeout(30):  # 30 segundos máximo
            updated_regrets, updated_strategy = _cfr_step_pure(
                regrets, 
                strategy, 
                key, 
                config,
                lut_keys,
                lut_values,
                lut_table_size
            )
        
        elapsed = time.time() - start_time
        print(f"✅ _cfr_step_pure completado en {elapsed:.2f} segundos")
        
        # Verificar resultados
        regrets_changed = not jnp.allclose(regrets, updated_regrets, atol=1e-6)
        strategy_changed = not jnp.allclose(strategy, updated_strategy, atol=1e-6)
        
        print(f"   Regrets actualizados: {regrets_changed}")
        print(f"   Strategy actualizada: {strategy_changed}")
        print(f"   Max regret: {jnp.max(jnp.abs(updated_regrets)):.6f}")
        
        if regrets_changed or strategy_changed:
            print("✅ FUNCIÓN FUNCIONA: Arrays se actualizaron correctamente")
            return True
        else:
            print("⚠️ ADVERTENCIA: Arrays no cambiaron (posible problema)")
            return False
            
    except TimeoutError as e:
        print(f"❌ TIMEOUT: {e}")
        print("   La función se colgó durante la compilación JIT del motor de juego")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alternative_approach():
    """Sugerir enfoque alternativo si la función pura falla"""
    print("\n🔧 ENFOQUE ALTERNATIVO: Mock del motor de juego")
    print("="*60)
    
    print("Si _cfr_step_pure falla con el motor real, podemos:")
    print("1. Crear un mock de unified_batch_simulation_with_lut")
    print("2. Probar la lógica CFR sin el motor pesado")
    print("3. Integrar gradualmente el motor real")
    
    # Ejemplo de mock simple
    try:
        import jax.numpy as jnp
        
        def mock_game_simulation(keys, lut_keys, lut_values, lut_table_size):
            """Mock simple del motor de juego"""
            batch_size = keys.shape[0]
            
            # Payoffs simulados
            payoffs = jnp.ones((batch_size, 6)) * 10.0
            
            # Historiales simulados  
            histories = jnp.zeros((batch_size, 60), dtype=jnp.int32)
            
            # Game results simulados
            game_results = {
                'hole_cards': jnp.ones((batch_size, 6, 2), dtype=jnp.int32) * 20,
                'final_community': jnp.ones((batch_size, 5), dtype=jnp.int32) * 10,
                'final_pot': jnp.ones(batch_size) * 100.0,
                'player_stacks': jnp.ones((batch_size, 6)) * 1000.0,
                'player_bets': jnp.ones((batch_size, 6)) * 15.0
            }
            
            return payoffs, histories, game_results
        
        # Probar el mock
        keys = jax.random.split(jax.random.PRNGKey(42), 2)
        lut_keys = jnp.array([0, 1], dtype=jnp.int32)
        lut_values = jnp.array([100, 200], dtype=jnp.int32)
        
        payoffs, histories, game_results = mock_game_simulation(keys, lut_keys, lut_values, 2)
        
        print(f"✅ Mock funciona: payoffs{payoffs.shape}, game_results keys: {list(game_results.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error en mock: {e}")
        return False

def main():
    """Ejecutar test aislado de la función CFR"""
    print("🎯 TEST AISLADO: _cfr_step_pure")
    print("Objetivo: Detectar exactamente dónde se produce el bloqueo")
    print("\n")
    
    # Test 1: Función pura minimal
    success1 = test_cfr_step_minimal()
    
    # Test 2: Enfoque alternativo
    success2 = test_alternative_approach()
    
    # Resumen y recomendaciones
    print("\n" + "="*60)
    print("📊 RESUMEN Y RECOMENDACIONES")
    print("="*60)
    
    if success1:
        print("🎉 ¡EXCELENTE! _cfr_step_pure funciona con configuración minimal")
        print("\n💡 RECOMENDACIÓN:")
        print("   - La integración funciona correctamente")
        print("   - Usar configuraciones pequeñas para tests iniciales")
        print("   - Incrementar gradualmente batch_size y max_info_sets")
        print("   - La compilación JIT inicial es lenta pero funciona")
        
    else:
        print("⚠️ _cfr_step_pure se bloquea con el motor de juego real")
        print("\n💡 SOLUCIONES RECOMENDADAS:")
        print("   1. INMEDIATA: Usar mock del motor para desarrollo inicial")
        print("   2. MEDIA: Optimizar unified_batch_simulation_with_lut")
        print("   3. LARGA: Investigar memory leaks en motor de juego")
        
        if success2:
            print("\n✅ Mock approach disponible como alternativa")
            
    return success1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)