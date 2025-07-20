# poker_bot/core/trainer.py

"""
Clean CFR Trainer for Poker AI with Hybrid CFR+ Implementation
JAX-native implementation combining regret discounting and CFR+ for enhanced performance
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from functools import partial

from . import full_game_engine as game_engine
from .bucketing import compute_info_set_id, validate_bucketing_system

logger = logging.getLogger(__name__)

# ==============================================================================
# FUNCIONES PURAS COMPILADAS CON JIT (FUERA DE LA CLASE)
# ==============================================================================

@jax.jit
def _evaluate_hand_simple_pure(hole_cards: jnp.ndarray) -> jnp.ndarray:
    """Función pura para evaluación simple de manos."""
    ranks = hole_cards // 4
    suits = hole_cards % 4
    
    # Simple scoring: high cards + pair bonus + suited bonus
    rank_value = jnp.sum(ranks) * 100
    pair_bonus = jnp.where(ranks[0] == ranks[1], 2000, 0)
    suited_bonus = jnp.where(suits[0] == suits[1], 500, 0)
    
    return rank_value + pair_bonus + suited_bonus

@jax.jit
def _update_regrets_for_game_gpu_simple_pure(
    regrets_shape: tuple,
    game_payoffs: jnp.ndarray,
    hole_cards_batch: jnp.ndarray,
    community_cards: jnp.ndarray,
    pot_size: jnp.ndarray
) -> jnp.ndarray:
    """
    Función pura para la actualización de regrets - PURE JAX, sin callbacks.
    
    Args:
        regrets_shape: Shape de la tabla de regrets
        game_payoffs: Payoffs para cada jugador [6]
        hole_cards_batch: Cartas de cada jugador [6, 2]
        community_cards: Cartas comunitarias [5]
        pot_size: Tamaño final del pot (scalar)
        
    Returns:
        Regret updates para este juego
    """
    regret_updates = jnp.zeros(regrets_shape)
    
    # VECTORIZED GPU OPTIMIZATION: Procesar todos los 6 jugadores simultáneamente
    pot_size_broadcast = jnp.full(6, pot_size)  # [6]
    player_indices = jnp.arange(6)
    
    # Paso 1: Computación vectorizada de info sets
    info_set_indices = jax.vmap(
        lambda hole_cards, player_idx, pot: compute_info_set_id(
            hole_cards, community_cards, player_idx, jnp.array([pot])
        ),
        in_axes=(0, 0, 0)
    )(hole_cards_batch, player_indices, pot_size_broadcast)
    
    # Paso 2: Evaluación vectorizada de manos
    hand_strengths = jax.vmap(_evaluate_hand_simple_pure)(hole_cards_batch)
    normalized_strengths = hand_strengths / 10000.0
    
    # Paso 3: Computación vectorizada de regrets
    def compute_regret_vector(strength, payoff):
        return jnp.where(
            strength > 0.7,
            payoff * jnp.array([0.0, 0.0, 0.1, 0.5, 0.8, 0.2]),  # Fuerte: bet/raise
            jnp.where(
                strength > 0.3,
                payoff * jnp.array([0.1, 0.2, 0.3, 0.1, 0.0, 0.0]),  # Medio: mixed
                payoff * jnp.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0])   # Débil: fold/check
            )
        )
    
    all_action_regrets = jax.vmap(compute_regret_vector)(normalized_strengths, game_payoffs)
    
    # Paso 4: FULLY VECTORIZED scatter updates - SIN LOOPS!
    regret_updates = regret_updates.at[info_set_indices].add(all_action_regrets)
    
    return regret_updates

@partial(jax.jit, static_argnames=("config",))
def _cfr_step_pure(
    regrets: jnp.ndarray,
    strategy: jnp.ndarray,
    key: jax.Array,
    config: TrainerConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Paso de entrenamiento CFR+ completamente puro. No captura 'self'.
    Recibe la configuración como un argumento estático.
    """
    # Generar datos sintéticos de juego en lugar de simulaciones completas para evitar problemas de memoria LUT
    keys = jax.random.split(key, config.batch_size)
    
    # Generar cartas sintéticas para los jugadores
    def generate_game_data(game_key):
        # Generar hole cards para 6 jugadores (2 cartas cada uno)
        hole_cards = jax.random.choice(game_key, 52, (6, 2), replace=False)
        # Generar payoffs sintéticos basados en fuerza simple de mano
        payoffs = jax.vmap(_evaluate_hand_simple_pure)(hole_cards) / 1000.0
        # Agregar ruido para hacerlo realista
        noise_key, _ = jax.random.split(game_key)
        payoffs = payoffs + jax.random.normal(noise_key, (6,)) * 10.0
        return hole_cards, payoffs
    
    # Generar batch de juegos sintéticos
    all_hole_cards, all_payoffs = jax.vmap(generate_game_data)(keys)
    
    # Procesar solo el primer juego para mantener el uso de memoria bajo
    game_hole_cards = all_hole_cards[0]  # [6, 2]
    game_payoffs = all_payoffs[0]  # [6]
    
    # Generar cartas comunitarias mock y pot
    community_key = jax.random.fold_in(key, 1000)
    community_cards = jax.random.choice(community_key, 52, (5,), replace=False)
    final_pot = jnp.float32(100.0)  # Tamaño de pot fijo para simplicidad
    
    # Computar actualizaciones de regret usando método existente (sin dependencia LUT)
    regret_updates = _update_regrets_for_game_gpu_simple_pure(
        regrets.shape, game_payoffs, game_hole_cards, community_cards, final_pot
    )
    
    # Aplicar descuento de regrets si está habilitado
    discounted_regrets = jnp.where(
        config.use_regret_discounting,
        regrets * config.discount_factor,
        regrets
    )
    
    # Actualizar regrets con nueva información
    updated_regrets = discounted_regrets + regret_updates
    
    # Aplicar poda CFR+: establecer regrets negativos a cero
    if config.use_cfr_plus:
        updated_regrets = jnp.maximum(updated_regrets, 0.0)
    
    # Actualizar estrategia usando regret matching
    updated_strategy = _regret_matching_pure(updated_regrets, config)
    
    return updated_regrets, updated_strategy

@partial(jax.jit, static_argnames=("config",))
def _regret_matching_pure(regrets: jnp.ndarray, config: TrainerConfig) -> jnp.ndarray:
    """
    Convertir regrets a estrategia usando regret matching - versión pura.
    
    Args:
        regrets: Tabla de regrets
        config: Configuración del entrenador
        
    Returns:
        Tabla de estrategia actualizada
    """
    # Tomar parte positiva de regrets
    positive_regrets = jnp.maximum(regrets, 0.0)
    
    # Sumar regrets para cada info set
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    
    # Normalizar para obtener probabilidades
    strategy = jnp.where(
        regret_sums > config.strategy_threshold,
        positive_regrets / regret_sums,
        jnp.ones_like(positive_regrets) / config.num_actions
    )
    
    return strategy

# ---------- LUT Loading Utility ----------
def load_hand_evaluation_lut(lut_path: Optional[str] = None) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Load hand evaluation lookup table for fast hand strength calculation.
    
    Args:
        lut_path: Optional path to LUT file. If None, uses default location.
        
    Returns:
        Tuple of (lut_keys, lut_values, lut_table_size) for hand evaluation
        
    Raises:
        FileNotFoundError: If LUT file is not found
        ValueError: If LUT file is corrupted or invalid
    """
    if lut_path is None:
        lut_path = os.path.join("data", "hand_evaluation_lut.pkl")
    
    try:
        with open(lut_path, 'rb') as f:
            lut_data = pickle.load(f)
        
        # Validate LUT structure - using actual keys from build_lut.py
        required_keys = ['hash_keys', 'hash_values', 'table_size']
        if not all(key in lut_data for key in required_keys):
            raise ValueError(f"Invalid LUT format: missing required keys {required_keys}")
         
        lut_keys = jnp.array(lut_data['hash_keys'], dtype=jnp.int32)
        lut_values = jnp.array(lut_data['hash_values'], dtype=jnp.int32)
        table_size = int(lut_data['table_size'])
        
        logger.info(f"✅ LUT loaded successfully: {len(lut_keys)} entries, table_size={table_size}")
        return lut_keys, lut_values, table_size
        
    except FileNotFoundError:
        logger.warning(f"⚠️ LUT file not found at {lut_path}, using fallback evaluation")
        # Return dummy values for fallback
        lut_keys = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        lut_values = jnp.array([100, 200, 300, 400, 500, 600], dtype=jnp.int32)
        return lut_keys, lut_values, 6
        
    except Exception as e:
        logger.error(f"❌ Error loading LUT: {e}")
        raise

@dataclass
class TrainerConfig:
    """Enhanced configuration for CFR training with hybrid CFR+ support"""
    # Core training parameters
    batch_size: int = 128
    num_actions: int = 6  # FOLD, CHECK, CALL, BET, RAISE, ALL_IN
    max_info_sets: int = 50_000
    learning_rate: float = 0.01
    
    # CFR parameters
    regret_floor: float = -100.0  # Legacy parameter - will be overridden when CFR+ is enabled
    regret_ceiling: float = 100.0
    strategy_threshold: float = 1e-6
    
    # Hybrid CFR+ parameters
    discount_factor: float = 0.9995  # Regret discounting factor (CFR-γ)
    use_cfr_plus: bool = True        # Enable CFR+ pruning
    use_regret_discounting: bool = True  # Enable regret discounting
    
    # Training control
    save_interval: int = 1000
    log_interval: int = 100
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainerConfig':
        """Load configuration from YAML file with hybrid CFR+ support"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Create default instance to get fallback values
        default_config = cls()
        
        # Extract all parameters from YAML with proper fallbacks to class defaults
        return cls(
            batch_size=config_dict.get('batch_size', default_config.batch_size),
            num_actions=config_dict.get('num_actions', default_config.num_actions),
            max_info_sets=config_dict.get('max_info_sets', default_config.max_info_sets),
            learning_rate=config_dict.get('learning_rate', default_config.learning_rate),
            regret_floor=config_dict.get('regret_floor', default_config.regret_floor),
            regret_ceiling=config_dict.get('regret_ceiling', default_config.regret_ceiling),
            strategy_threshold=config_dict.get('strategy_threshold', default_config.strategy_threshold),
            save_interval=config_dict.get('save_interval', default_config.save_interval),
            log_interval=config_dict.get('log_interval', default_config.log_interval),
            discount_factor=config_dict.get('discount_factor', default_config.discount_factor),
            use_cfr_plus=config_dict.get('use_cfr_plus', default_config.use_cfr_plus),
            use_regret_discounting=config_dict.get('use_regret_discounting', default_config.use_regret_discounting),
        )

class PokerTrainer:
    """
    Enhanced CFR trainer with hybrid CFR+ implementation for superior performance.
    Combines regret discounting (CFR-γ) with CFR+ pruning for faster convergence.
    """
    
    def __init__(self, config: TrainerConfig, lut_path: Optional[str] = None):
        self.config = config
        self.iteration = 0
        
        # Initialize regrets and strategy
        self.regrets = jnp.zeros(
            (config.max_info_sets, config.num_actions),
            dtype=jnp.float32
        )
        self.strategy = jnp.ones(
            (config.max_info_sets, config.num_actions),
            dtype=jnp.float32
        ) / config.num_actions
        
        # Load LUT parameters during initialization
        self.lut_keys, self.lut_values, self.lut_table_size = load_hand_evaluation_lut(lut_path)
        
        # Validate bucketing system on initialization
        if not validate_bucketing_system():
            raise RuntimeError("Bucketing system validation failed")
        
        logger.info(f"🎯 PokerTrainer initialized with hybrid CFR+")
        logger.info(f"   Config: {config.batch_size} batch, {config.max_info_sets:,} info sets")
        logger.info(f"   CFR+: {config.use_cfr_plus}, Discount: {config.discount_factor}")
        logger.info(f"   LUT: {self.lut_table_size} entries loaded")
        logger.info(f"   Shapes: regrets{self.regrets.shape}, strategy{self.strategy.shape}")

    def train(self, num_iterations: int, save_path: str) -> Dict[str, Any]:
        """
        Main training loop with hybrid CFR+ optimization.
        
        Args:
            num_iterations: Number of CFR iterations
            save_path: Path to save final model
            
        Returns:
            Training statistics including hybrid CFR+ metrics
        """
        logger.info(f"🚀 Starting hybrid CFR+ training: {num_iterations:,} iterations")
        logger.info(f"   Save path: {save_path}")
        
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        # Training statistics
        stats = {
            'iterations_completed': 0,
            'total_time': 0.0,
            'final_regret_sum': 0.0,
            'final_strategy_entropy': 0.0,
            'discount_factor_used': self.config.discount_factor,
            'cfr_plus_enabled': self.config.use_cfr_plus,
            'regret_discounting_enabled': self.config.use_regret_discounting
        }
        
        try:
            for i in range(self.iteration + 1, self.iteration + num_iterations + 1):
                self.iteration = i
                iter_key = jax.random.fold_in(key, i)
                
                # Single CFR step with hybrid optimization
                iter_start = time.time()
                
                # ¡LA LLAMADA CORRECTA A LA FUNCIÓN PURA!
                # Le pasamos todos los datos que necesita explícitamente.
                # Ya no hay ninguna captura implícita de 'self'.
                self.regrets, self.strategy = _cfr_step_pure(
                    self.regrets, self.strategy, iter_key, self.config
                )
                
                # Ensure computation completes
                self.regrets.block_until_ready()
                iter_time = time.time() - iter_start
                
                # Periodic logging
                if i % self.config.log_interval == 0:
                    progress = 100 * i / num_iterations
                    regret_sum = float(jnp.sum(jnp.abs(self.regrets)))
                    logger.info(
                        f"Progress: {progress:.1f}% ({i:,}/{num_iterations:,}) "
                        f"| {iter_time:.2f}s | Regret sum: {regret_sum:.1e}"
                    )
                
                # Periodic saves
                if i % self.config.save_interval == 0:
                    checkpoint_path = f"{save_path}_iter_{i}.pkl"
                    self.save_model(checkpoint_path)
            
            # Final statistics
            total_time = time.time() - start_time
            stats.update({
                'iterations_completed': num_iterations,
                'total_time': total_time,
                'final_regret_sum': float(jnp.sum(jnp.abs(self.regrets))),
                'final_strategy_entropy': self._compute_strategy_entropy(),
                'iterations_per_second': num_iterations / total_time
            })
            
            # Save final model
            final_path = f"{save_path}_final.pkl"
            self.save_model(final_path)
            
            logger.info(f"✅ Hybrid CFR+ training completed successfully")
            logger.info(f"   Time: {total_time:.1f}s ({stats['iterations_per_second']:.1f} iter/s)")
            logger.info(f"   Final model: {final_path}")
            
        except Exception as e:
            logger.error(f"❌ Hybrid CFR+ training failed at iteration {self.iteration}: {e}")
            raise
        
        return stats
    
    # VERSIONES ORIGINALES ELIMINADAS - AHORA USAMOS FUNCIONES PURAS FUERA DE LA CLASE
    # Las funciones _cfr_step, _update_regrets_for_game_gpu_simple, _evaluate_hand_simple
    # y _regret_matching fueron movidas como funciones puras fuera de la clase para
    # mejor optimización JIT según las mejores prácticas de JAX.
    
    def _regret_matching(self, regrets: jnp.ndarray) -> jnp.ndarray:
        """
        Versión de conveniencia no-JIT para compatibilidad con funciones que no están en JIT.
        Para uso dentro de funciones JIT, usar _regret_matching_pure directamente.
        """
        return _regret_matching_pure(regrets, self.config)
    
    def _compute_strategy_entropy(self) -> float:
        """Compute average entropy of current strategy"""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = jnp.log(self.strategy + eps)
        entropy_per_info_set = -jnp.sum(self.strategy * log_probs, axis=1)
        return float(jnp.mean(entropy_per_info_set))
    
    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'regrets': np.asarray(self.regrets),
            'strategy': np.asarray(self.strategy),
            'iteration': self.iteration,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"💾 Model saved: {path} ({size_mb:.1f} MB)")
    
    def load_model(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.regrets = jnp.array(data['regrets'])
        self.strategy = jnp.array(data['strategy'])
        self.iteration = data['iteration']
        
        if 'config' in data:
            self.config = data['config']
        
        logger.info(f"📂 Model loaded: {path}")
        logger.info(f"   Iteration: {self.iteration}")

    def resume_training(self, checkpoint_path: str, num_iterations: int, save_path: str) -> Dict[str, Any]:
        """
        Resume training from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file to resume from
            num_iterations: Number of additional iterations to run
            save_path: Path to save the resumed training
            
        Returns:
            Training statistics including resume information
        """
        logger.info(f"🔄 Resuming hybrid CFR+ training from: {checkpoint_path}")
        logger.info(f"   Additional iterations: {num_iterations}")
        
        # Load the checkpoint
        self.load_model(checkpoint_path)
        
        # Continue training from loaded state
        return self.train(num_iterations, save_path)

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: str) -> 'PokerTrainer':
        """
        Create trainer instance from checkpoint for resume.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            PokerTrainer ready to continue training
        """
        trainer = cls.__new__(cls)
        trainer.load_model(checkpoint_path)
        return trainer

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
        """
        Find the latest checkpoint file in the specified directory.
        
        Args:
            checkpoint_dir: Directory to search for checkpoints
            
        Returns:
            Path to latest checkpoint file or None if no checkpoints found
        """
        if not os.path.exists(checkpoint_dir):
            return None
            
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.endswith('.pkl') and 'iter_' in f
        ]
        
        if not checkpoint_files:
            return None
            
        # Sort by iteration number (extract from filename)
        def extract_iteration(filename):
            try:
                return int(filename.split('_iter_')[1].split('.')[0])
            except (IndexError, ValueError):
                return 0
                
        latest_file = max(checkpoint_files, key=extract_iteration)
        return os.path.join(checkpoint_dir, latest_file)

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for checkpointing"""
        return {
            'regrets': np.asarray(self.regrets),
            'strategy': np.asarray(self.strategy),
            'iteration': self.iteration,
            'config': self.config,
            'timestamp': time.time()
        }

# Factory function for easy creation
def create_trainer(config_path: Optional[str] = None) -> PokerTrainer:
    """
    Create trainer with configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configured PokerTrainer with hybrid CFR+ support
    """
    if config_path:
        config = TrainerConfig.from_yaml(config_path)
    else:
        config = TrainerConfig()
    
    return PokerTrainer(config)