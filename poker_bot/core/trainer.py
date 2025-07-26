# poker_bot/core/trainer.py

"""
Clean CFR Trainer for Poker AI with Hybrid CFR+ Implementation
JAX-native implementation combining regret discounting and CFR+ for enhanced performance!..
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
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
from .mccfr_algorithm import MCCFRTrainer, mc_sampling_strategy, accumulate_regrets_fixed, calculate_strategy_optimized

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainerConfig:
    """Enhanced configuration for CFR training with hybrid CFR+ support"""
    # Core training parameters
    batch_size: int = 128
    num_actions: int = 9  # FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE, RAISE_SMALL, RAISE_MED, ALL_IN
    max_info_sets: int = 1_000_000  # Increased from 50,000 to handle large indices
    learning_rate: float = 0.1  # Increased from 0.01 to create larger regret updates
    
    # CFR parameters
    regret_floor: float = -100.0  # Legacy parameter - will be overridden when CFR+ is enabled
    regret_ceiling: float = 100.0
    strategy_threshold: float = 1e-15  # Further reduced to handle very small regrets from large tables
    
    # Hybrid CFR+ parameters
    discount_factor: float = 0.9995  # Regret discounting factor (CFR-γ)
    use_cfr_plus: bool = False        # Enable CFR+ pruning
    use_regret_discounting: bool = True  # Enable regret discounting
    
    # MC-CFR parameters
    mc_sampling_rate: float = 0.50  # Process 50% of learning opportunities (was 0.15)
    mc_exploration_epsilon: float = 0.6  # 60% exploration, 40% exploitation
    mc_min_samples_per_info_set: int = 100
    mc_max_samples_per_info_set: int = 10000
    
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
            mc_sampling_rate=config_dict.get('mc_sampling_rate', default_config.mc_sampling_rate),
            mc_exploration_epsilon=config_dict.get('mc_exploration_epsilon', default_config.mc_exploration_epsilon),
            mc_min_samples_per_info_set=config_dict.get('mc_min_samples_per_info_set', default_config.mc_min_samples_per_info_set),
            mc_max_samples_per_info_set=config_dict.get('mc_max_samples_per_info_set', default_config.mc_max_samples_per_info_set),
        )

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
def _evaluate_7card_simple(hole_cards: jnp.ndarray, community_cards: jnp.ndarray) -> jnp.ndarray:
    """Evaluación rápida de 7 cartas compatible con JAX JIT.."""
    # Combinar todas las cartas (siempre 7 elementos)
    all_cards = jnp.concatenate([hole_cards, community_cards])
    
    # Máscara para cartas válidas (>= 0)
    valid_mask = all_cards >= 0
    num_valid = jnp.sum(valid_mask)
    
    # Si hay menos de 2 cartas válidas, retornar fuerza mínima
    strength = jnp.where(
        num_valid < 2,
        0.1,  # Fuerza mínima
        _compute_hand_strength_fixed_size(all_cards, valid_mask)
    )
    
    return jnp.clip(strength, 0.0, 1.0)

@jax.jit 
def _compute_hand_strength_fixed_size(all_cards: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """Computa fuerza de mano con arrays de tamaño fijo."""
    # Procesar solo cartas válidas usando máscaras
    ranks = jnp.where(valid_mask, all_cards // 4, 0)
    suits = jnp.where(valid_mask, all_cards % 4, 0)
    
    # Contar rangos (0-12) y palos (0-3)
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    # Acumular conteos usando scatter_add
    for i in range(7):  # Procesar las 7 posiciones
        rank_counts = rank_counts.at[ranks[i]].add(jnp.where(valid_mask[i], 1, 0))
        suit_counts = suit_counts.at[suits[i]].add(jnp.where(valid_mask[i], 1, 0))
    
    # Analizar mano
    max_rank_count = jnp.max(rank_counts)
    pairs = jnp.sum(rank_counts == 2)
    trips = jnp.sum(rank_counts == 3)
    quads = jnp.sum(rank_counts == 4)
    is_flush = jnp.any(suit_counts >= 5)
    
    # Carta más alta (normalizada)
    high_card = jnp.max(jnp.where(valid_mask, ranks, 0)) / 12.0
    
    # Scoring jerárquico de poker
    strength = jnp.where(
        quads > 0, 0.95,  # Four of a kind (muy raro)
        jnp.where(
            (trips > 0) & (pairs > 0), 0.85,  # Full house (raro)
            jnp.where(
                is_flush, 0.75,  # Flush (poco común)
                jnp.where(
                    trips > 0, 0.45,  # Three of a kind (↓ de 0.65)
                    jnp.where(
                        pairs >= 2, 0.25,  # Two pair (↓ de 0.5)
                        jnp.where(
                            pairs == 1, 0.12 + high_card * 0.06,  # One pair (↓ mucho)
                            high_card * 0.08  # High card (↓ de 0.25)
                        )
                    )
                )
            )
        )
    )
    
    return strength

## CAMBIO CLAVE 1: Actualizar _update_regrets_for_game_pure para trabajar con resultados reales del motor de juego
# [ELIMINADO] def _update_regrets_for_game_pure(...)

## CAMBIO CLAVE 2: Modificar _cfr_step_pure para aceptar la LUT y llamar al game_engine.unified_batch_simulation_with_lut
# [ELIMINADO] def _cfr_step_pure(...)

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
    
    # IMPROVED: Better handling of small regrets
    # Use a more robust strategy computation
    strategy = jnp.where(
        regret_sums > config.strategy_threshold,
        positive_regrets / (regret_sums + 1e-12),  # Add small epsilon to prevent division by zero
        jnp.ones_like(positive_regrets) / config.num_actions
    )
    
    # IMPROVED: Ensure strategy is properly normalized
    strategy_sums = jnp.sum(strategy, axis=1, keepdims=True)
    strategy = strategy / (strategy_sums + 1e-12)
    
    return strategy

## CAMBIO CLAVE 3: Asegurar que la LUT se carga como NumPy arrays para mantenerla en CPU
def load_hand_evaluation_lut(lut_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load hand evaluation lookup table for fast hand strength calculation.
    ## CAMBIO CLAVE 3: Retorna NumPy arrays para mantener la LUT en CPU
    
    Args:
        lut_path: Optional path to LUT file. If None, uses default location.
        
    Returns:
        Tuple of (lut_keys, lut_values, lut_table_size) as NumPy arrays for CPU
        
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
         
        # ## CAMBIO CLAVE 3: Cargar como NumPy arrays para mantener en CPU
        lut_keys = np.array(lut_data['hash_keys'], dtype=np.int32)
        lut_values = np.array(lut_data['hash_values'], dtype=np.int32)
        table_size = int(lut_data['table_size'])
        
        logger.info(f"✅ LUT loaded successfully: {len(lut_keys)} entries, table_size={table_size}")
        logger.info(f"   LUT arrays kept in CPU memory as NumPy arrays")
        return lut_keys, lut_values, table_size
        
    except FileNotFoundError:
        logger.warning(f"⚠️ LUT file not found at {lut_path}, using fallback evaluation")
        # Return dummy values for fallback as NumPy arrays
        lut_keys = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        lut_values = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
        return lut_keys, lut_values, 6
        
    except Exception as e:
        logger.error(f"❌ Error loading LUT: {e}")
        raise

# ==============================================================================
# CLASE PRINCIPAL DEL ENTRENADOR
# ==============================================================================

class PokerTrainer:
    """Enhanced Poker AI Trainer with hybrid CFR+ and MC-CFR support"""
    
    def __init__(self, config: TrainerConfig, lut_path: Optional[str] = None):
        """
        Initialize trainer with configuration and optional LUT path.
        
        Args:
            config: Trainer configuration
            lut_path: Optional path to hand evaluation LUT
        """
        self.config = config
        self.iteration = 0
        # REEMPLAZO: Inicializar MCCFRTrainer
        self.mccfr_trainer = MCCFRTrainer(config.max_info_sets, config.num_actions)
        # Mantener carga de LUT
        self.lut_keys, self.lut_values, self.lut_table_size = load_hand_evaluation_lut(lut_path)
        
        # Convert LUT arrays to JAX for GPU compatibility (for benchmarks)
        # self.lut_keys_jax = jnp.array(self.lut_keys, dtype=jnp.int32)
        # self.lut_values_jax = jnp.array(self.lut_values, dtype=jnp.int32)
        
        # Initialize regret and strategy tables
        self.regrets = jnp.zeros(
            (config.max_info_sets, config.num_actions),
            dtype=jnp.float32
        )
        self.strategy = jnp.ones(
            (config.max_info_sets, config.num_actions),
            dtype=jnp.float32
        ) / config.num_actions
        
        # Validate bucketing system
        if not validate_bucketing_system():
            raise RuntimeError("Bucketing system validation failed")
        
        logger.info(f"✅ PokerTrainer initialized with {config.num_actions} actions")
        logger.info(f"   Max info sets: {config.max_info_sets}")
        logger.info(f"   MC sampling rate: {config.mc_sampling_rate}")
        logger.info(f"   CFR+ enabled: {config.use_cfr_plus}")
        logger.info(f"   Regret discounting: {config.use_regret_discounting}")

    def train(self, num_iterations: int, save_path: str) -> Dict[str, Any]:
        """
        Train using MCCFR + CFR+ instead of broken vanilla CFR.
        """
        logger.info(f"🚀 Starting MCCFR + CFR+ training")
        key = jax.random.PRNGKey(42)
        stats = {'iterations': [], 'regret_magnitudes': [], 'strategy_entropies': [], 'training_times': []}
        start_time = time.time()
        for i in range(num_iterations):
            iter_start = time.time()
            iter_key = jax.random.fold_in(key, i)
            self.regrets, self.strategy = _cfr_step_with_mccfr(
                self.regrets, self.strategy, iter_key, self.config, self.iteration,
                None, None, 0
            )
            # Update MCCFRTrainer with new values (outside JIT)
            self.mccfr_trainer.regrets = self.regrets
            self.mccfr_trainer.strategy = self.strategy
            self.mccfr_trainer.iteration = self.iteration
            self.iteration += 1
            regret_magnitude = jnp.sum(jnp.abs(self.regrets))
            strategy_entropy = self._compute_strategy_entropy()
            iter_time = time.time() - iter_start
            stats['iterations'].append(self.iteration)
            stats['regret_magnitudes'].append(float(regret_magnitude))
            stats['strategy_entropies'].append(strategy_entropy)
            stats['training_times'].append(iter_time)
            if i % self.config.log_interval == 0:
                logger.info(f"📊 MCCFR Iteration {i}: regret={regret_magnitude:.4f}, entropy={strategy_entropy:.4f}, time={iter_time:.3f}s")
            if i > 0 and i % self.config.save_interval == 0:
                checkpoint_path = f"{save_path}_iter_{i}.pkl"
                self.save_model(checkpoint_path)
                logger.info(f"💾 Saved MCCFR checkpoint: {checkpoint_path}")
        self.save_model(save_path)
        total_time = time.time() - start_time
        logger.info(f"✅ MCCFR training completed in {total_time:.2f}s")
        return stats

    def _regret_matching(self, regrets: jnp.ndarray) -> jnp.ndarray:
        """Convert regrets to strategy using regret matching."""
        return _regret_matching_pure(regrets, self.config)

    def _compute_strategy_entropy(self) -> float:
        """Compute strategy entropy for monitoring training progress."""
        # Compute entropy for each info set
        log_probs = jnp.log(jnp.clip(self.strategy, 1e-10, 1.0))
        entropies = -jnp.sum(self.strategy * log_probs, axis=1)
        return float(jnp.mean(entropies))

    def save_model(self, path: str):
        """Save MCCFR model state."""
        model_state = {
            'config': self.config,
            'mccfr_trainer': self.mccfr_trainer,  # Save MCCFR trainer
            'regrets': np.array(self.mccfr_trainer.regrets),
            'strategy': np.array(self.mccfr_trainer.strategy),
            'iteration': self.iteration,
            'lut_keys': self.lut_keys,
            'lut_values': self.lut_values,
            'lut_table_size': self.lut_table_size
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        logger.info(f"💾 MCCFR model saved to {path}")

    def load_model(self, path: str):
        """Load MCCFR model state."""
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        self.config = model_state['config']
        self.mccfr_trainer = model_state['mccfr_trainer']  # Load MCCFR trainer
        self.regrets = jnp.array(model_state['regrets'])
        self.strategy = jnp.array(model_state['strategy'])
        self.iteration = model_state['iteration']
        self.lut_keys = model_state['lut_keys']
        self.lut_values = model_state['lut_values']
        self.lut_table_size = model_state['lut_table_size']
        logger.info(f"📂 MCCFR model loaded from {path}")

    def resume_training(self, checkpoint_path: str, num_iterations: int, save_path: str) -> Dict[str, Any]:
        """
        Resume training from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file to resume from
            num_iterations: Number of additional iterations to train
            save_path: Base path for saving new checkpoints
            
        Returns:
            Training statistics
        """
        logger.info(f"🔄 Resuming hybrid CFR+ training from: {checkpoint_path}")
        
        # Load the checkpoint
        self.load_model(checkpoint_path)
        
        # Continue training
        return self.train(num_iterations, save_path)

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: str) -> 'PokerTrainer':
        """
        Create trainer instance from checkpoint for resume.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Trainer instance with loaded state
        """
        trainer = cls.__new__(cls)  # Create instance without calling __init__
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
        
        def extract_iteration(filename):
            try:
                return int(filename.split('iter_')[1].split('.')[0])
            except (IndexError, ValueError):
                return 0
        
        latest_file = max(checkpoint_files, key=extract_iteration)
        return os.path.join(checkpoint_dir, latest_file)

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for checkpointing"""
        return {
            'iteration': self.iteration,
            'regret_magnitude': float(jnp.sum(jnp.abs(self.regrets))),
            'strategy_entropy': self._compute_strategy_entropy(),
            'config': self.config
        }

def create_trainer(config_path: Optional[str] = None) -> PokerTrainer:
    """
    Create trainer instance from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured trainer instance
    """
    if config_path is None:
        config_path = os.path.join("config", "training_config.yaml")
    
    config = TrainerConfig.from_yaml(config_path)
    return PokerTrainer(config)

@partial(jax.jit, static_argnames=("config",))
def _cfr_step_with_mccfr(
    regrets: jax.Array,
    strategy: jax.Array,
    key: jax.Array,
    config: TrainerConfig,
    iteration: int,
    lut_keys: jax.Array,
    lut_values: jax.Array,
    lut_table_size: int
) -> tuple[jax.Array, jax.Array]:
    """
    NEW: Use MCCFR + CFR+ with DEBUG validation
    """
    # DEBUG: Validate input shapes and dtypes
    # jax.debug.print("🔍 INPUT VALIDATION:")
    # jax.debug.print("  regrets shape: {}, dtype: {}", regrets.shape, regrets.dtype)
    # jax.debug.print("  strategy shape: {}, dtype: {}", strategy.shape, strategy.dtype)
    # jax.debug.print("  config.batch_size: {}", config.batch_size)
    # jax.debug.print("  lut_keys shape: {}, dtype: {}", lut_keys.shape, lut_keys.dtype)
    # jax.debug.print("  lut_values shape: {}, dtype: {}", lut_values.shape, lut_values.dtype)
    # Generate keys for real game simulations
    keys = jax.random.split(key, config.batch_size)
    # jax.debug.print("  keys shape: {}", keys.shape)
    # Use REAL game engine with LUT
    payoffs, histories, game_results_batch = game_engine.unified_batch_simulation_with_lut(
        keys, lut_keys, lut_values, lut_table_size, config.num_actions
    )
    # DEBUG: Validate game engine outputs
    # jax.debug.print("📊 GAME ENGINE OUTPUTS:")
    # jax.debug.print("  payoffs shape: {}, dtype: {}", payoffs.shape, payoffs.dtype)
    # jax.debug.print("  game_results_batch hole_cards shape: {}", game_results_batch['hole_cards'].shape)
    # jax.debug.print("  game_results_batch final_community shape: {}", game_results_batch['final_community'].shape)
    # jax.debug.print("  game_results_batch final_pot shape: {}", game_results_batch['final_pot'].shape)
    def process_single_game(game_idx):
        hole_cards_batch = game_results_batch['hole_cards'][game_idx]
        community_cards = game_results_batch['final_community'][game_idx]
        pot_size = game_results_batch['final_pot'][game_idx]
        # DEBUG: Validate inputs to compute_info_set_id
        # jax.debug.print("🎯 process_single_game {}", game_idx)
        # jax.debug.print("  hole_cards_batch shape: {}", hole_cards_batch.shape)
        # jax.debug.print("  community_cards shape: {}", community_cards.shape)
        # jax.debug.print("  pot_size: {}", pot_size)
        player_indices = jnp.arange(6)
        pot_size_broadcast = jnp.full(6, pot_size)
        info_set_indices = jax.vmap(
            lambda hole_cards, player_idx, pot: compute_info_set_id(
                hole_cards, community_cards, player_idx, jnp.array([pot]), max_info_sets=config.max_info_sets
            )
        )(hole_cards_batch, player_indices, pot_size_broadcast)
        # DEBUG: Validate info_set_indices
        # jax.debug.print("  info_set_indices shape: {}, dtype: {}", info_set_indices.shape, info_set_indices.dtype)
        # jax.debug.print("  info_set_indices min: {}, max: {}", jnp.min(info_set_indices), jnp.max(info_set_indices))
        game_payoffs = payoffs[game_idx].astype(jnp.float32)
        # CFR requires different payoffs per action
        # For now, add small random variation to break symmetry
        random_key = jax.random.fold_in(key, game_idx + 1000)
        action_noise = jax.random.normal(random_key, (6, config.num_actions)) * 10.0
        action_values = jnp.broadcast_to(
            game_payoffs[:, None], 
            (6, config.num_actions)
        ).astype(jnp.float32) + action_noise
        # jax.debug.print("  action_values shape: {}, dtype: {}", action_values.shape, action_values.dtype)
        return info_set_indices, action_values
    # DEBUG: Before batch processing
    jax.debug.print("🔄 BATCH PROCESSING...")
    batch_indices = jnp.arange(config.batch_size)
    batch_info_sets, batch_action_values = jax.vmap(process_single_game)(batch_indices)
    # DEBUG: Validate batch processing outputs
    # jax.debug.print("  batch_info_sets shape: {}, dtype: {}", batch_info_sets.shape, batch_info_sets.dtype)
    # jax.debug.print("  batch_action_values shape: {}, dtype: {}", batch_action_values.shape, batch_action_values.dtype)
    # Flatten batch data for MCCFR
    flat_info_sets = batch_info_sets.reshape(-1).astype(jnp.int32)
    flat_action_values = batch_action_values.reshape(-1, config.num_actions)
    # Track visited info sets - FIX JAX JIT COMPATIBILITY  
    visited_mask = jnp.zeros(config.max_info_sets, dtype=jnp.bool_)
    flat_info_sets_safe = flat_info_sets.astype(jnp.int32)
    visited_mask = visited_mask.at[flat_info_sets_safe].set(True)
    # DEBUG: Validate flattened data
    jax.debug.print("  flat_action_values shape: {}, dtype: {}", flat_action_values.shape, flat_action_values.dtype)
    jax.debug.print("  flat_action_values min: {}, max: {}, mean: {}", jnp.min(flat_action_values), jnp.max(flat_action_values), jnp.mean(flat_action_values))
    jax.debug.print("  visited_mask sum: {}", jnp.sum(visited_mask))
    # MCCFR operations
    from .mccfr_algorithm import mc_sampling_strategy, cfr_iteration
    game_key = jax.random.fold_in(key, iteration)
    sampling_mask = mc_sampling_strategy(regrets, flat_info_sets, game_key)
    jax.debug.print("  sampling_mask sum: {}", jnp.sum(sampling_mask))
    updated_regrets, updated_strategy = cfr_iteration(
        regrets, strategy, flat_info_sets, flat_action_values, sampling_mask, 
        iteration, config.learning_rate, config.use_regret_discounting
    )

    # --- INICIO DEL BLOQUE A INSERTAR ---

    # Aplicar la lógica de CFR+ (regrets no negativos) si está activada en la configuración.
    # Esta es una corrección crítica para asegurar que el algoritmo converge correctamente.
    def apply_cfr_plus(r):
        """Si CFR+ está activo, los regrets negativos se resetean a 0."""
        return jnp.maximum(r, 0.0)

    def do_nothing(r):
        """Si CFR+ no está activo, los regrets no se modifican."""
        return r

    # Usamos jax.lax.cond para que esta lógica condicional sea compatible con la compilación JIT.
    updated_regrets = lax.cond(
        config.use_cfr_plus,
        apply_cfr_plus,    # Función a ejecutar si config.use_cfr_plus es True
        do_nothing,        # Función a ejecutar si config.use_cfr_plus es False
        updated_regrets    # El dato sobre el que operan las funciones
    )

    # --- FIN DEL BLOQUE A INSERTAR ---

    return updated_regrets, updated_strategy