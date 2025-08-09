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
from typing import Dict, Any, Optional, List
from functools import partial

from . import full_game_engine as game_engine
from .full_game_engine import GameState
from .bucketing import compute_info_set_id_enhanced, compute_info_set_id_enhanced as compute_info_set_id, validate_bucketing_system
from .mccfr_algorithm import MCCFRTrainer, mc_sampling_strategy, accumulate_regrets_fixed, calculate_strategy_optimized
from .starting_hands import classify_starting_hand, classify_starting_hand_with_position
from .config import TrainerConfig

# CFR Counterfactual Simulation Functions
# Remove the broken helper functions
# def copy_game_state(game_state):
#     """Create a deep copy of the game state for simulation."""
#     return {
#         'hole_cards': game_state['hole_cards'].copy(),
#         'community_cards': game_state['community_cards'].copy(),
#         'pot_size': game_state['pot_size'].copy(),
#         'payoffs': game_state['payoffs'].copy()
#     }

# def apply_action_to_state(simulated_state, player_idx, action_idx):
#     """Apply action with hand-strength awareness."""
#     base_payoff = simulated_state['payoffs'][player_idx]
    
#     # Evaluar fuerza de mano del jugador
#     hole_cards = simulated_state['hole_cards'][player_idx]
#     community_cards = simulated_state['community_cards']
#     # Usar player_idx como posición (0-5 para 6-max)
#     hand_strength = _evaluate_7card_simple(hole_cards, community_cards, player_idx)
    
#     # Lógica específica por fuerza de mano
#     if action_idx == 0:  # FOLD
#         # FOLD es bueno con manos débiles, malo con manos fuertes
#         fold_value = jnp.where(hand_strength < 0.3, base_payoff + 1000, base_payoff - 1500)
#         simulated_state['payoffs'] = simulated_state['payoffs'].at[player_idx].set(fold_value)
#     elif action_idx in [3,4,5,6,7,8]:  # Aggressive actions  
#         # Agresividad es buena con manos fuertes, mala con manos débiles
#         aggro_value = jnp.where(hand_strength > 0.7, base_payoff + 1500, base_payoff - 1000)
#         simulated_state['payoffs'] = simulated_state['payoffs'].at[player_idx].set(aggro_value)
#     else:  # CHECK/CALL
#         # Penalizar check/call con manos muy débiles
#         neutral_value = jnp.where(
#             hand_strength < 0.2, base_payoff - 800,  # Malo con trash
#             base_payoff + jax.random.normal(jax.random.PRNGKey(42), ()) * 25
#         )
#         simulated_state['payoffs'] = simulated_state['payoffs'].at[player_idx].set(neutral_value)
    
#     return simulated_state

# def simulate_game_to_completion(simulated_state, player_idx):
#     """Simulate the game to completion and return the final payoff for the player."""
#     # For now, return the modified payoff from the action
#     # This can be enhanced with full game simulation
#     return simulated_state['payoffs'][player_idx]

logger = logging.getLogger(__name__)



# ==============================================================================
# FUNCIONES PURAS COMPILADAS CON JIT (FUERA DE LA CLASE)!
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
def _evaluate_7card_simple(hole_cards: jnp.ndarray, community_cards: jnp.ndarray, position: int = 2) -> jnp.ndarray:
    """Evaluación multi-street completa."""
    from .starting_hands import evaluate_hand_strength_multi_street
    
    return evaluate_hand_strength_multi_street(hole_cards, community_cards, position)

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
        rank_idx = ranks[i].astype(jnp.int32)
        suit_idx = suits[i].astype(jnp.int32)
        rank_counts = rank_counts.at[rank_idx].add(jnp.where(valid_mask[i], 1, 0))
        suit_counts = suit_counts.at[suit_idx].add(jnp.where(valid_mask[i], 1, 0))
    
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
        logger.info(f"🚀 Starting MCCFR + CFR+ training with robust PRNGKey handling")
        key = jax.random.PRNGKey(int(time.time()))  # Usar el tiempo para una semilla siempre nueva
        stats = {'iterations': [], 'regret_magnitudes': [], 'strategy_entropies': [], 'training_times': []}
        start_time = time.time()
        for i in range(num_iterations):
            iter_start = time.time()
            
            # --- NUEVA LÓGICA DE ALEATORIEDAD ---
            # En lugar de derivar, generamos una clave completamente nueva para cada iteración.
            # Esto rompe cualquier posible correlación entre batches.
            key, iter_key = jax.random.split(key) 
            # ------------------------------------
            
            self.regrets, self.strategy = _cfr_step_with_mccfr(
                self.regrets, self.strategy, iter_key, self.config, self.iteration,
                self.lut_keys, self.lut_values, self.lut_table_size
            )
            

            
            # Update MCCFRTrainer with new values (outside JIT)
            self.mccfr_trainer.regrets = self.regrets
            self.mccfr_trainer.strategy = self.strategy
            self.mccfr_trainer.iteration = self.iteration
            self.iteration += 1
            regret_magnitude = jnp.sum(jnp.abs(self.regrets))
            entropy_details = self._compute_strategy_entropy_detailed()
            strategy_entropy = entropy_details['overall_entropy']
            iter_time = time.time() - iter_start
            stats['iterations'].append(self.iteration)
            stats['regret_magnitudes'].append(float(regret_magnitude))
            stats['strategy_entropies'].append(strategy_entropy)
            stats['training_times'].append(iter_time)
            if i % self.config.log_interval == 0:
                logger.info(f"📊 MCCFR Iteration {i}: regret={regret_magnitude:.4f}, entropy={strategy_entropy:.4f}, trained_sets={entropy_details['trained_info_sets']}, time={iter_time:.3f}s")
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

    def _compute_strategy_entropy_detailed(self) -> Dict[str, float]:
        """Compute detailed entropy statistics for debugging."""
        # Entropy for each info set
        log_probs = jnp.log(jnp.clip(self.strategy, 1e-10, 1.0))
        entropies = -jnp.sum(self.strategy * log_probs, axis=1)
        
        # Find which info sets have non-uniform strategies (been trained)
        uniform_entropy = jnp.log(self.config.num_actions)  # ln(9) ≈ 2.197
        is_trained = jnp.abs(entropies - uniform_entropy) > 0.05  # Threshold for "different from uniform"
        

        
        return {
            'overall_entropy': float(jnp.mean(entropies)),
            'trained_info_sets': int(jnp.sum(is_trained)),
            'total_info_sets': int(self.config.max_info_sets),
            'trained_percentage': float(jnp.sum(is_trained) / self.config.max_info_sets * 100),
            'trained_entropy_avg': float(jnp.mean(jnp.where(is_trained, entropies, jnp.nan), axis=0, where=~jnp.isnan(entropies))),
            'uniform_entropy': float(uniform_entropy)
        }

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

def generate_diverse_game_state(key: jax.Array, num_players: int = 6) -> GameState:
    """
    Genera un estado de juego MUCHO MÁS diverso (V2 - PRNGKey Fix).
    Cada operación aleatoria usa una sub-clave única para garantizar la independencia.
    """
    # --- Manejo Correcto de Claves Aleatorias ---
    keys = jax.random.split(key, 8) # Necesitamos 8 subclaves, una para cada decisión
    k_deck, k_street, k_stack_scen, k_stack_mult, k_pot_scen, k_pot_val, k_player, k_final = keys
    
    # --- Generación de Estado (Lógica sin cambios, solo claves diferentes) ---
    deck = jax.random.permutation(k_deck, jnp.arange(52))
    street = jax.random.randint(k_street, (), 0, 4)
    hole_cards = deck[:num_players*2].reshape(num_players, 2)
    
    num_community_cards = jnp.where(street == 1, 3, jnp.where(street == 2, 4, jnp.where(street == 3, 5, 0)))
    potential_community_cards = lax.dynamic_slice(deck, (num_players*2,), (5,))
    mask = jnp.arange(5) < num_community_cards
    community_cards = jnp.where(mask, potential_community_cards, -1)

    # Diversidad en Stacks
    stack_scenarios = jax.random.randint(k_stack_scen, (), 0, 5)
    base_stacks = lax.switch(
        stack_scenarios,
        [
            lambda: jnp.full((num_players,), 1000.0), # Normal
            lambda: jnp.array([50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]), # Mixto
            lambda: jnp.array([10000.0, 8000.0, 6000.0, 4000.0, 2000.0, 1000.0]), # Deep
            lambda: jnp.full((num_players,), 100.0), # Todos Short
            lambda: jnp.full((num_players,), 5000.0) # Todos Deep
        ]
    )
    stack_multipliers = jax.random.uniform(k_stack_mult, shape=(num_players,), minval=0.1, maxval=2.0)
    stacks = base_stacks * stack_multipliers
    
    # Diversidad en Pot Sizes
    pot_scenarios = jax.random.randint(k_pot_scen, (), 0, 4)
    pot = lax.switch(
        pot_scenarios,
        [
            lambda: jax.random.uniform(k_pot_val, shape=(), minval=15.0, maxval=100.0), # Small
            lambda: jax.random.uniform(k_pot_val, shape=(), minval=100.0, maxval=500.0), # Medium
            lambda: jax.random.uniform(k_pot_val, shape=(), minval=500.0, maxval=2000.0), # Large
            lambda: jax.random.uniform(k_pot_val, shape=(), minval=2000.0, maxval=10000.0) # Massive
        ]
    )
    
    # Jugador Actual Aleatorio
    current_player = jax.random.randint(k_player, (), 0, num_players)
    
    return GameState(
        stacks=stacks,
        bets=jnp.zeros((num_players,)),
        player_status=jnp.zeros((num_players,), dtype=jnp.int8),
        hole_cards=hole_cards,
        comm_cards=community_cards,
        cur_player=jnp.array([current_player], dtype=jnp.int8),
        street=street[None].astype(jnp.int8),
        pot=pot[None],
        deck=deck,
        deck_ptr=jnp.array([num_players*2 + num_community_cards]),
        acted_this_round=jnp.zeros((num_players,), dtype=jnp.int8),
        key=k_final, # Usamos la última subclave para el estado final
        action_hist=jnp.zeros((60,), dtype=jnp.int32),
        hist_ptr=jnp.array([0])
    )

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
    Versión final con exploración forzada y corrección de formas para JIT.
    """
    keys = jax.random.split(key, config.batch_size)

    def generate_and_play_batch(k):
        # Desacoplamos la aleatoriedad: una llave para la decisión, otra para el juego.
        decision_key, game_key = jax.random.split(k)
        
        # FORZAR EXPLORACIÓN: 70% de las veces empezamos desde un estado diverso (turn, river, etc.)
        do_full_game_sim = jax.random.uniform(decision_key) < 0.3

        return lax.cond(
            do_full_game_sim,
            # Pasamos la llave 'game_key' a la simulación.
            lambda: game_engine.play_one_game(
                game_key, lut_keys, lut_values, lut_table_size, config.num_actions
            ),
            # Pasamos la llave 'game_key' a la simulación.
            lambda: game_engine.play_from_state(
                generate_diverse_game_state(game_key), lut_keys, lut_values, lut_table_size, config.num_actions
            )
        )

    payoffs, histories, game_results_batch = jax.vmap(generate_and_play_batch)(keys)
    
    def process_single_game(game_idx):
        # Trajectory data
        info_hist = game_results_batch['info_hist'][game_idx]
        legal_hist = game_results_batch['legal_hist'][game_idx]
        player_hist = game_results_batch['player_hist'][game_idx]
        pot_hist = game_results_batch['pot_hist'][game_idx]
        comm_hist = game_results_batch['comm_hist'][game_idx]
        hist_len = game_results_batch['hist_len'][game_idx]
        actions_hist = histories[game_idx][:hist_len]

        # Truncate to actual decisions
        decision_idx = jnp.arange(hist_len)
        info_ids = info_hist[:hist_len]
        legal_mask = legal_hist[:hist_len]
        players = player_hist[:hist_len]
        # Payoffs per player for this terminal outcome
        terminal_payoffs = payoffs[game_idx]

        # Build outcome-sampling Q(s,a): only chosen action gets terminal payoff for acting player; others 0
        def per_decision_values(i):
            legal = legal_mask[i]
            chosen = actions_hist[i]
            player_idx = players[i].astype(jnp.int32)
            payoff = terminal_payoffs[player_idx]
            values = jnp.zeros((config.num_actions,), dtype=jnp.float32)
            values = values.at[chosen].set(payoff.astype(jnp.float32))
            values = jnp.where(legal, values, 0.0)
            return values

        per_values = jax.vmap(per_decision_values)(decision_idx)
        return info_ids, per_values

    batch_info_sets, batch_action_values = jax.vmap(process_single_game)(jnp.arange(config.batch_size))
    # Asegurar tipos y aplanado por nodos
    flat_info_sets = batch_info_sets.reshape(-1).astype(jnp.int32)
    flat_action_values = batch_action_values.reshape(-1, config.num_actions)

    from .mccfr_algorithm import mc_sampling_strategy, cfr_iteration
    game_key = jax.random.fold_in(key, iteration)
    
    sampling_mask = mc_sampling_strategy(regrets, flat_info_sets, game_key, config)
    
    updated_regrets, updated_strategy = cfr_iteration(
        regrets, strategy, flat_info_sets, flat_action_values, sampling_mask, 
        iteration, config.learning_rate, config.use_regret_discounting, config
    )

    def apply_cfr_plus(r):
        return jnp.maximum(r, 0.0)
    def do_nothing(r):
        return r

    updated_regrets = lax.cond(
        config.use_cfr_plus,
        apply_cfr_plus,
        do_nothing,
        updated_regrets
    )

    return updated_regrets, updated_strategy