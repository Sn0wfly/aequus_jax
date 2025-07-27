#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for Post-Optimization Validation (Updated for MCCFR)
========================================================================================

Validates critical performance improvements from optimizations.
- Tests the current MCCFR-based training step.
- Measures iterations per second after JIT compilation.
"""

import time
import logging
import psutil
import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Añade el path para importar desde poker_bot
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer
from poker_bot.core.config import TrainerConfig
from poker_bot.core.trainer import _cfr_step_with_mccfr # Importa la nueva función

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResults:
    """Store comprehensive benchmark results"""
    iterations_per_second: float
    total_time: float
    avg_step_time: float
    jit_compilation_time: float
    memory_peak_mb: float
    improvement_factor: float
    meets_expected_improvement: bool

class PerformanceBenchmark:
    """Comprehensive benchmark for post-optimization validation"""

    def __init__(self):
        self.baseline_iter_per_sec = 1.25
        self.expected_min_improvement = 10
        self.iterations_to_test = 100

    def run_comprehensive_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark suite"""
        logger.info("🎯 COMPREHENSIVE PERFORMANCE BENCHMARK - (MCCFR VALIDATION)")
        logger.info("=" * 80)

        process = psutil.Process(os.getpid())
        
        logger.info("📊 PHASE 1: Initializing Trainer and Data...")
        trainer, config, key, lut_data = self._initialize_trainer()
        
        logger.info("📊 PHASE 2: JIT Compilation Warm-up...")
        start_jit = time.time()
        # Llama a la nueva función para compilar
        trainer.regrets, trainer.strategy = _cfr_step_with_mccfr(
            trainer.regrets, trainer.strategy, key, config, 0,
            lut_data['keys'], lut_data['values'], lut_data['size']
        )
        trainer.regrets.block_until_ready() # Asegura que la compilación termine
        jit_time = time.time() - start_jit
        logger.info(f"   ✅ JIT compilation finished in {jit_time:.2f}s")
        
        logger.info("📊 PHASE 3: Performance Benchmark (100 iterations)...")
        perf_metrics = self._run_performance_test(trainer, config, key, lut_data)

        memory_peak = process.memory_info().rss / (1024 * 1024)
        
        improvement_factor = perf_metrics['iter_per_sec'] / self.baseline_iter_per_sec
        
        results = BenchmarkResults(
            iterations_per_second=perf_metrics['iter_per_sec'],
            total_time=perf_metrics['total_time'],
            avg_step_time=perf_metrics['avg_step_time'],
            jit_compilation_time=jit_time,
            memory_peak_mb=memory_peak,
            improvement_factor=improvement_factor,
            meets_expected_improvement=perf_metrics['iter_per_sec'] >= (self.baseline_iter_per_sec * self.expected_min_improvement)
        )

        self._generate_report(results)
        return results

    def _initialize_trainer(self) -> Tuple[Any, Any, Any, Dict]:
        """Initialize trainer for testing"""
        config = TrainerConfig(
            batch_size=128,
            max_info_sets=1000,
            log_interval=10
        )
        trainer = PokerTrainer(config)
        key = jax.random.PRNGKey(42)
        
        # Carga los datos de LUT como arrays de JAX para la prueba
        lut_data = {
            'keys': jnp.array(trainer.lut_keys),
            'values': jnp.array(trainer.lut_values),
            'size': trainer.lut_table_size
        }

        logger.info(f"   ✅ Trainer initialized: {config.batch_size} batch size")
        return trainer, config, key, lut_data

    def _run_performance_test(self, trainer, config, key, lut_data) -> Dict[str, Any]:
        """Run 100-iteration performance benchmark"""
        step_times = []
        total_start = time.time()

        for i in range(self.iterations_to_test):
            step_start = time.time()
            iter_key = jax.random.fold_in(key, i + 1)
            
            # USA LA NUEVA FUNCIÓN
            trainer.regrets, trainer.strategy = _cfr_step_with_mccfr(
                trainer.regrets, trainer.strategy, iter_key, config, i,
                lut_data['keys'], lut_data['values'], lut_data['size']
            )
            trainer.regrets.block_until_ready()
            
            step_time = time.time() - step_start
            step_times.append(step_time)

            if (i + 1) % 20 == 0:
                logger.info(f"     Progress: {i+1}/{self.iterations_to_test} ({step_time:.3f}s)")

        total_time = time.time() - total_start
        
        avg_step_time = np.mean(step_times)
        iter_per_sec = self.iterations_to_test / total_time

        logger.info(f"   ✅ Performance Results:")
        logger.info(f"      Total time: {total_time:.2f}s")
        logger.info(f"      Iterations/sec: {iter_per_sec:.2f}")

        return {
            'total_time': total_time,
            'iter_per_sec': iter_per_sec,
            'avg_step_time': avg_step_time,
        }

    def _generate_report(self, results: BenchmarkResults):
        """Generate comprehensive benchmark report"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE BENCHMARK RESULTS (MCCFR)")
        logger.info("=" * 80)
        
        logger.info(f"\n🚀 PERFORMANCE SUMMARY:")
        logger.info(f"   Baseline (pre-optimization): {self.baseline_iter_per_sec:.1f} iter/s")
        logger.info(f"   Current (post-optimization): {results.iterations_per_second:.1f} iter/s")
        logger.info(f"   Improvement Factor: {results.improvement_factor:.1f}x")
        logger.info(f"   Meets Expected (10-100x): {'✅ YES' if results.meets_expected_improvement else '❌ NO'}")

        logger.info(f"\n⏱️ DETAILED METRICS:")
        logger.info(f"   JIT Compilation Time: {results.jit_compilation_time:.2f}s")
        logger.info(f"   Total Benchmark Time: {results.total_time:.2f}s")
        logger.info(f"   Average Step Time: {results.avg_step_time:.4f}s")

        logger.info(f"\n💾 MEMORY USAGE:")
        logger.info(f"   Memory Peak: {results.memory_peak_mb:.1f} MB")
        
        logger.info("=" * 80)

def run_benchmark():
    """Run the comprehensive performance benchmark"""
    benchmark = PerformanceBenchmark()
    return benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    try:
        results = run_benchmark()
        print(f"\nBenchmark completed! Improvement factor: {results.improvement_factor:.1f}x")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise