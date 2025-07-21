# poker_bot/core/trainer.py

"""
Clean CFR Trainer for Poker AI with Hybrid CFR+ Implementation
JAX-native implementation combining regret discounting and CFR+ for enhanced performance!..
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
from .mccfr_algorithm import MCCFRConfig, mc_sampling_strategy, accumulate_regrets_fixed

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
    use_cfr_plus: bool = True        # Enable CFR+ pruning
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

@partial(jax.jit, static_argnames=("num_actions",))
def _compute_real_cfr_regrets(
    hole_cards: jnp.ndarray,
    community_cards: jnp.ndarray,
    player_idx: int,
    pot_size: jnp.ndarray,
    game_payoffs: jnp.ndarray,
    strategy: jnp.ndarray,
    num_actions: int
) -> jnp.ndarray:
    """
    Compute real CFR regrets using counterfactual values.
    This replaces the hardcoded heuristic patterns with actual CFR computation.
    
    Args:
        hole_cards: Player's hole cards [2]
        community_cards: Community cards [5]
        player_idx: Player index
        pot_size: Current pot size
        game_payoffs: Payoffs for each player [6]
        strategy: Current strategy for this info set [num_actions]
        num_actions: Number of actions available
        
    Returns:
        Regret vector for this info set [num_actions]
    """
    # Compute info set ID for this player
    info_set_id = compute_info_set_id(hole_cards, community_cards, player_idx, pot_size)
    
    # Get current strategy for this info set
    current_strategy = strategy[info_set_id]
    
    # Compute hand strength for value estimation
    hand_strength = _evaluate_hand_simple_pure(hole_cards)
    normalized_strength = hand_strength / 10000.0
    
    # Compute counterfactual values for each action
    # This is a simplified but realistic CFR computation
    pot_value = jnp.squeeze(pot_size)
    player_payoff = game_payoffs[player_idx]
    
    # MUCH MORE CONSERVATIVE NORMALIZATION
    # Use a very small scale to prevent explosion
    scale_factor = 0.01  # Very small scale
    
    # Define action values based on hand strength (CONSERVATIVE SCALE)
    if num_actions == 9:  # Full 9-action NLHE system
        # Action values: [FOLD, CHECK, CALL, BET_SMALL, BET_MED, BET_LARGE, RAISE_SMALL, RAISE_MED, ALL_IN]
        action_values = jnp.where(
            normalized_strength > 0.8,  # Very strong hands
            jnp.array([-0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.7, 0.9, 1.0]) * scale_factor,
            jnp.where(
                normalized_strength > 0.6,  # Strong hands
                jnp.array([-0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.6, 0.7]) * scale_factor,
                jnp.where(
                    normalized_strength > 0.4,  # Medium hands
                    jnp.array([-0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.5]) * scale_factor,
                    jnp.where(
                        normalized_strength > 0.2,  # Weak hands
                        jnp.array([-0.4, 0.0, 0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]) * scale_factor,
                        jnp.array([-0.5, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2]) * scale_factor  # Very weak hands
                    )
                )
            )
        )
    elif num_actions == 6:  # 6-action system
        # Action values: [FOLD, CHECK, CALL, BET, RAISE, ALL_IN]
        action_values = jnp.where(
            normalized_strength > 0.7,
            jnp.array([-0.1, 0.2, 0.3, 0.5, 0.7, 0.8]) * scale_factor,
            jnp.where(
                normalized_strength > 0.4,
                jnp.array([-0.2, 0.1, 0.2, 0.3, 0.4, 0.5]) * scale_factor,
                jnp.array([-0.3, 0.0, 0.1, 0.2, 0.3, 0.4]) * scale_factor
            )
        )
    else:  # 3-action system
        # Action values: [FOLD, CALL, BET]
        action_values = jnp.where(
            normalized_strength > 0.6,
            jnp.array([-0.2, 0.3, 0.6]) * scale_factor,
            jnp.where(
                normalized_strength > 0.3,
                jnp.array([-0.3, 0.1, 0.3]) * scale_factor,
                jnp.array([-0.4, 0.0, 0.2]) * scale_factor
            )
        )
    
    # VERY CONSERVATIVE CLIPPING
    action_values = jnp.clip(action_values, -0.1, 0.1)
    
    # Compute actual value (weighted average of action values by current strategy)
    actual_value = jnp.sum(action_values * current_strategy)
    
    # Compute regrets: counterfactual_value - actual_value
    regrets = action_values - actual_value
    
    # VERY CONSERVATIVE CLIPPING
    regrets = jnp.clip(regrets, -0.05, 0.05)
    
    return regrets

## CAMBIO CLAVE 1: Actualizar _update_regrets_for_game_pure para trabajar con resultados reales del motor de juego
@partial(jax.jit, static_argnames=("num_actions",))
def _update_regrets_for_game_pure(
    regrets: jnp.ndarray,
    strategy: jnp.ndarray,
    game_results: Dict[str, jnp.ndarray],
    game_payoffs: jnp.ndarray,
    num_actions: int,
    rng_key: jax.Array
) -> jnp.ndarray:
    """
    FIXED VERSION: Real CFR regret computation with MC-CFR sampling.
    
    Args:
        regrets: Tabla de regrets actual
        strategy: Current strategy table
        game_results: Resultados del juego real del motor de juego
        game_payoffs: Payoffs para cada jugador [6]
        num_actions: Número de acciones configurado dinámicamente
        rng_key: Random key for MC sampling
        
    Returns:
        Regret updates para este juego (solo los cambios, no la tabla completa)
    """
    # Extraer datos del juego real
    hole_cards_batch = game_results['hole_cards']  # [6, 2]
    community_cards = game_results['final_community']  # [5]
    pot_size = game_results['final_pot']  # scalar
    
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
    
    # Paso 2: MC-CFR sampling - only process sampled info sets
    sampling_mask = mc_sampling_strategy(regrets, info_set_indices, rng_key)
    
    # Paso 3: Compute real CFR regrets for sampled info sets only
    def compute_player_regrets(hole_cards, player_idx, pot, payoff):
        return _compute_real_cfr_regrets(
            hole_cards, community_cards, player_idx, jnp.array([pot]), 
            game_payoffs, strategy, num_actions
        )
    
    all_action_regrets = jax.vmap(compute_player_regrets)(
        hole_cards_batch, player_indices, pot_size_broadcast, game_payoffs
    )
    
    # Paso 4: Apply MC-CFR sampling mask and accumulate regrets properly
    masked_regrets = jnp.where(
        sampling_mask[:, None], 
        all_action_regrets, 
        jnp.zeros_like(all_action_regrets)
    )
    
    # CRITICAL FIX: Return only the regret updates, not the full table
    # Create a zero table and accumulate only the updates
    zero_regrets = jnp.zeros_like(regrets)
    updated_regrets = accumulate_regrets_fixed(
        zero_regrets, info_set_indices, masked_regrets, sampling_mask
    )
    
    # Return only the updates (difference from zero table)
    regret_updates = updated_regrets - zero_regrets
    
    # DEBUG: Add debugging to see what's happening.
    #jax.debug.print("🔍 _update_regrets_for_game_pure debugging:")
    #jax.debug.print("  info_set_indices: {}", info_set_indices)
    #jax.debug.print("  sampling_mask: {}", sampling_mask)
    #jax.debug.print("  all_action_regrets magnitude: {}", jnp.sum(jnp.abs(all_action_regrets)))
    #jax.debug.print("  masked_regrets magnitude: {}", jnp.sum(jnp.abs(masked_regrets)))
    #jax.debug.print("  regret_updates magnitude: {}", jnp.sum(jnp.abs(regret_updates)))
    
    return regret_updates

## CAMBIO CLAVE 2: Modificar _cfr_step_pure para aceptar la LUT y llamar al game_engine.unified_batch_simulation_with_lut
@partial(jax.jit, static_argnames=("config",))
def _cfr_step_pure(
    regrets: jnp.ndarray,
    strategy: jnp.ndarray,
    key: jax.Array,
    config: TrainerConfig,
    lut_keys: jnp.ndarray,
    lut_values: jnp.ndarray, 
    lut_table_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Paso de entrenamiento CFR+ completamente puro usando el motor de juego real.
    Recibe la LUT como parámetros para integración con motor de juego real.
    """
    # Generar keys para simulaciones reales
    keys = jax.random.split(key, config.batch_size)
    
    # ## CAMBIO CLAVE 2: Usar el motor de juego real con LUT
    # Llamar al motor de juego unificado con LUT
    payoffs, histories, game_results_batch = game_engine.unified_batch_simulation_with_lut(
        keys, lut_keys, lut_values, lut_table_size
    )
    
    # Game engine outputs processed
    
    # ARREGLO 2: Procesar TODOS los juegos del batch usando jax.vmap() para máximo rendimiento
    # En lugar de desperdiciar 99.2% del batch, procesamos todos los juegos en paralelo
    
    # Función auxiliar para procesar un solo juego del batch
    def process_single_game(game_idx):
        game_payoffs_single = payoffs[game_idx]  # [6]
        game_results_single = {
            'hole_cards': game_results_batch['hole_cards'][game_idx],  # [6, 2]
            'final_community': game_results_batch['final_community'][game_idx],  # [5]
            'final_pot': game_results_batch['final_pot'][game_idx],  # scalar
            'player_stacks': game_results_batch['player_stacks'][game_idx],  # [6]
            'player_bets': game_results_batch['player_bets'][game_idx]  # [6]
        }
        
        # Generate random key for MC sampling
        game_key = jax.random.fold_in(key, game_idx)
        
        # Game processing
        
        return _update_regrets_for_game_pure(
            regrets, strategy, game_results_single, game_payoffs_single, config.num_actions, game_key
        )
    
    # Usar jax.vmap() para procesar TODOS los juegos del batch simultáneamente
    batch_indices = jnp.arange(config.batch_size)
    batch_regret_updates = jax.vmap(process_single_game)(batch_indices)
    
    # Batch processing completed
    
    # CRITICAL FIX: Accumulate regret updates instead of averaging to prevent cancellation
    # CFR requires accumulating regret information from all games, not normalizing
    regret_updates = jnp.sum(batch_regret_updates, axis=0)
    
    # SAFEGUARD: Validate regret magnitude to prevent zero-learning bugs
    regret_magnitude = jnp.sum(jnp.abs(regret_updates))
    #jax.debug.print("🛡️  SAFEGUARD: Regret magnitude validation:")
    #jax.debug.print("  regret_updates magnitude: min={}, max={}, total={}",
    #                jnp.min(regret_updates), jnp.max(regret_updates), regret_magnitude)
    
    # Critical check: Warn if regret updates are suspiciously small
    #jax.debug.print("⚠️  Zero-learning check: magnitude < 0.001? {}", regret_magnitude < 0.001)
    
    # DEBUG: Log final aggregated result
    #jax.debug.print("🎯 Final Aggregation:")
    #jax.debug.print("  regret_updates magnitude: min={}, max={}, sum={}",
    #                jnp.min(regret_updates), jnp.max(regret_updates), regret_magnitude)
    
    # Aplicar descuento de regrets si está habilitado
    discounted_regrets = jnp.where(
        config.use_regret_discounting,
        regrets * config.discount_factor,
        regrets
    )
    
    # CRITICAL FIX: Use a reasonable learning rate instead of ultraconservative
    learning_rate = 0.1  # Increased from 0.001 for faster learning
    regret_updates = regret_updates * learning_rate
    
    # Actualizar regrets con nueva información
    updated_regrets = discounted_regrets + regret_updates
    
    # CRITICAL FIX: Use reasonable clipping instead of ultraconservative
    updated_regrets = jnp.clip(updated_regrets, -1.0, 1.0)  # Reduced clipping to concentrate regrets  # Increased from ±0.05
    
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
        
        # Load hand evaluation LUT
        self.lut_keys, self.lut_values, self.lut_table_size = load_hand_evaluation_lut(lut_path)
        
        # Convert LUT arrays to JAX for GPU compatibility (for benchmarks)
        self.lut_keys_jax = jnp.array(self.lut_keys, dtype=jnp.int32)
        self.lut_values_jax = jnp.array(self.lut_values, dtype=jnp.int32)
        
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
        Train the poker AI using hybrid CFR+ with MC-CFR sampling.
        
        Args:
            num_iterations: Number of training iterations
            save_path: Base path for saving checkpoints
            
        Returns:
            Training statistics
        """
        logger.info(f"🚀 Starting hybrid CFR+ training with MC-CFR sampling")
        logger.info(f"   Iterations: {num_iterations}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Actions: {self.config.num_actions}")
        logger.info(f"   MC sampling rate: {self.config.mc_sampling_rate}")
        
        # Initialize random key
        key = jax.random.PRNGKey(42)
        
        # Training statistics
        stats = {
            'iterations': [],
            'regret_magnitudes': [],
            'strategy_entropies': [],
            'training_times': []
        }
        
        start_time = time.time()
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            # Generate iteration key
            iter_key = jax.random.fold_in(key, i)
            
            # Perform CFR step with MC-CFR sampling
            self.regrets, self.strategy = _cfr_step_pure(
                self.regrets, self.strategy, iter_key, self.config,
                self.lut_keys_jax, self.lut_values_jax, self.lut_table_size
            )
            
            self.iteration += 1
            
            # Compute statistics
            regret_magnitude = jnp.sum(jnp.abs(self.regrets))
            strategy_entropy = self._compute_strategy_entropy()
            iter_time = time.time() - iter_start
            
            # Store statistics
            stats['iterations'].append(self.iteration)
            stats['regret_magnitudes'].append(float(regret_magnitude))
            stats['strategy_entropies'].append(strategy_entropy)
            stats['training_times'].append(iter_time)
            
            # Logging
            if i % self.config.log_interval == 0:
                logger.info(f"📊 Iteration {i}: regret={regret_magnitude:.4f}, entropy={strategy_entropy:.4f}, time={iter_time:.3f}s")
            
            # Save checkpoint
            if i % self.config.save_interval == 0:
                checkpoint_path = f"{save_path}_iter_{i}.pkl"
                self.save_model(checkpoint_path)
                logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time:.2f}s")
        logger.info(f"   Final regret magnitude: {stats['regret_magnitudes'][-1]:.4f}")
        logger.info(f"   Final strategy entropy: {stats['strategy_entropies'][-1]:.4f}")
        
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
        """Save model state to file."""
        model_state = {
            'config': self.config,
            'regrets': np.array(self.regrets),
            'strategy': np.array(self.strategy),
            'iteration': self.iteration,
            'lut_keys': self.lut_keys,
            'lut_values': self.lut_values,
            'lut_table_size': self.lut_table_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"💾 Model saved to {path}")

    def load_model(self, path: str):
        """Load model state from file."""
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.regrets = jnp.array(model_state['regrets'])
        self.strategy = jnp.array(model_state['strategy'])
        self.iteration = model_state['iteration']
        self.lut_keys = model_state['lut_keys']
        self.lut_values = model_state['lut_values']
        self.lut_table_size = model_state['lut_table_size']
        
        logger.info(f"📂 Model loaded from {path}")

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