# poker_bot/core/trainer.py

"""
Clean CFR Trainer for Poker AI
JAX-native implementation focused purely on CFR training logic
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

@dataclass
class TrainerConfig:
    """Clean configuration for CFR training"""
    # Core training parameters
    batch_size: int = 128
    num_actions: int = 6  # FOLD, CHECK, CALL, BET, RAISE, ALL_IN
    max_info_sets: int = 50_000
    learning_rate: float = 0.01
    
    # CFR parameters
    regret_floor: float = -100.0
    regret_ceiling: float = 100.0
    strategy_threshold: float = 1e-6
    
    # Training control
    save_interval: int = 1000
    log_interval: int = 100
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainerConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract relevant parameters
        return cls(
            batch_size=config_dict.get('batch_size', 128),
            learning_rate=config_dict.get('learning_rate', 0.01),
            # Add other parameters as needed
        )

class PokerTrainer:
    """
    Clean CFR trainer focused purely on training logic.
    All debugging, validation, and evaluation moved to separate modules.
    """
    
    def __init__(self, config: TrainerConfig):
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
        
        # Validate bucketing system on initialization
        if not validate_bucketing_system():
            raise RuntimeError("Bucketing system validation failed")
        
        logger.info(f"🎯 PokerTrainer initialized")
        logger.info(f"   Config: {config.batch_size} batch, {config.max_info_sets:,} info sets")
        logger.info(f"   Shapes: regrets{self.regrets.shape}, strategy{self.strategy.shape}")
    
    def train(self, num_iterations: int, save_path: str) -> Dict[str, Any]:
        """
        Main training loop - clean and focused.
        
        Args:
            num_iterations: Number of CFR iterations
            save_path: Path to save final model
            
        Returns:
            Training statistics
        """
        logger.info(f"🚀 Starting CFR training: {num_iterations:,} iterations")
        logger.info(f"   Save path: {save_path}")
        
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        # Training statistics
        stats = {
            'iterations_completed': 0,
            'total_time': 0.0,
            'final_regret_sum': 0.0,
            'final_strategy_entropy': 0.0
        }
        
        try:
            for i in range(1, num_iterations + 1):
                self.iteration = i
                iter_key = jax.random.fold_in(key, i)
                
                # Single CFR step
                iter_start = time.time()
                self.regrets, self.strategy = self._cfr_step(
                    self.regrets, self.strategy, iter_key
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
            
            logger.info(f"✅ Training completed successfully")
            logger.info(f"   Time: {total_time:.1f}s ({stats['iterations_per_second']:.1f} iter/s)")
            logger.info(f"   Final model: {final_path}")
            
        except Exception as e:
            logger.error(f"❌ Training failed at iteration {self.iteration}: {e}")
            raise
        
        return stats
    
    @partial(jax.jit, static_argnums=(0,))
    def _cfr_step(self, regrets: jnp.ndarray, strategy: jnp.ndarray, 
                  key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single CFR training step - JIT compiled for performance.
        
        Args:
            regrets: Current regret table
            strategy: Current strategy table  
            key: Random key for game simulation
            
        Returns:
            Updated (regrets, strategy) tuple
        """
        # Generate batch of game simulations
        keys = jax.random.split(key, self.config.batch_size)
        payoffs, game_results = self._simulate_games(keys)
        
        # Process each game in the batch
        def process_game(game_idx):
            return self._update_regrets_for_game(
                regrets, payoffs[game_idx], game_results, game_idx
            )
        
        # Vectorized regret updates
        batch_regret_updates = jax.vmap(process_game)(
            jnp.arange(self.config.batch_size)
        )
        
        # Accumulate regrets from all games
        updated_regrets = regrets + jnp.sum(batch_regret_updates, axis=0)
        
        # Clip regrets to prevent explosion
        clipped_regrets = jnp.clip(
            updated_regrets, 
            self.config.regret_floor, 
            self.config.regret_ceiling
        )
        
        # Update strategy using regret matching
        updated_strategy = self._regret_matching(clipped_regrets)
        
        return clipped_regrets, updated_strategy
    
    @partial(jax.jit, static_argnums=(0,))
    def _simulate_games(self, keys: jnp.ndarray) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Simulate batch of poker games.
        
        Args:
            keys: Random keys for each game
            
        Returns:
            (payoffs, game_results) tuple
        """
        # Use the clean game engine
        payoffs, histories, game_results = game_engine.unified_batch_simulation(keys)
        return payoffs, game_results
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_regrets_for_game(self, regrets: jnp.ndarray, game_payoffs: jnp.ndarray,
                                game_results: Dict[str, jnp.ndarray], 
                                game_idx: int) -> jnp.ndarray:
        """
        Update regrets for a single game.
        
        Args:
            regrets: Current regret table
            game_payoffs: Payoffs for each player
            game_results: Game simulation results
            game_idx: Index of game in batch
            
        Returns:
            Regret updates for this game
        """
        regret_updates = jnp.zeros_like(regrets)
        
        # VECTORIZED GPU OPTIMIZATION: Process all 6 players simultaneously
        all_hole_cards = game_results['hole_cards'][game_idx]  # [6, 2]  
        community_cards = game_results['final_community'][game_idx]  # [5]
        pot_size_broadcast = jnp.full(6, game_results['final_pot'][game_idx])  # [6]
        player_indices = jnp.arange(6)
        
        # MASSIVE GPU PARALLELIZATION: 6 players computed at once
        # This replaces the CPU loop with pure GPU vectorized operations
        
        # Step 1: Vectorized info set computation
        info_set_indices = jax.vmap(
            lambda hole_cards, player_idx, pot: compute_info_set_id(
                hole_cards, community_cards, player_idx, jnp.array([pot])
            ),
            in_axes=(0, 0, 0)
        )(all_hole_cards, player_indices, pot_size_broadcast)
        
        # Step 2: Vectorized hand evaluation
        hand_strengths = jax.vmap(self._evaluate_hand_simple)(all_hole_cards)
        normalized_strengths = hand_strengths / 10000.0
        
        # Step 3: Vectorized regret computation based on hand strength
        def compute_regret_vector(strength, payoff):
            return jnp.where(
                strength > 0.7,
                payoff * jnp.array([0.0, 0.0, 0.1, 0.5, 0.8, 0.2]) / 100.0,  # Strong: bet/raise
                jnp.where(
                    strength > 0.3, 
                    payoff * jnp.array([0.1, 0.2, 0.3, 0.1, 0.0, 0.0]) / 100.0,  # Medium: mixed
                    payoff * jnp.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0]) / 100.0   # Weak: fold/check
                )
            )
        
        all_action_regrets = jax.vmap(compute_regret_vector)(normalized_strengths, game_payoffs)
        
        # Step 4: Scatter updates (this part still sequential but minimal)
        for i in range(6):
            regret_updates = regret_updates.at[info_set_indices[i]].add(all_action_regrets[i])
        
        return regret_updates
    
# _extract_game_state removed - no longer needed with direct array passing
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_action_regrets(self, player_payoff: float, game_results: Dict[str, jnp.ndarray],
                               game_idx: int, player_idx: int) -> jnp.ndarray:
        """
        Compute regret for each action given the game outcome.
        
        Args:
            player_payoff: Payoff for this player
            game_results: Game results
            game_idx: Game index
            player_idx: Player index
            
        Returns:
            Array of regrets for each action
        """
        # Simplified regret computation
        # In practice, this would use counterfactual values
        
        hole_cards = game_results['hole_cards'][game_idx, player_idx]
        hand_strength = self._evaluate_hand_simple(hole_cards)
        
        # Normalize hand strength to [0, 1]
        normalized_strength = hand_strength / 10000.0
        
        # Compute value for each action
        action_values = jnp.array([
            # FOLD: Always 0, but might avoid losses
            jnp.where(player_payoff < 0, -player_payoff * 0.1, -1.0),
            # CHECK: Conservative play
            player_payoff * (0.5 + normalized_strength * 0.3),
            # CALL: Match aggression
            player_payoff * (0.6 + normalized_strength * 0.3),
            # BET: Aggressive with good hands
            player_payoff * (0.4 + normalized_strength * 0.6),
            # RAISE: Very aggressive
            player_payoff * (0.3 + normalized_strength * 0.7),
            # ALL_IN: Maximum aggression
            player_payoff * (0.2 + normalized_strength * 0.8)
        ])
        
        # Expected value is the actual payoff
        expected_value = player_payoff
        
        # Regret = action_value - expected_value
        regrets = action_values - expected_value
        
        return jnp.clip(regrets, -10.0, 10.0)
    
    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_hand_simple(self, hole_cards: jnp.ndarray) -> jnp.ndarray:
        """
        Simple hand evaluation for regret computation.
        
        Args:
            hole_cards: [2] array of card indices
            
        Returns:
            Simple hand strength estimate
        """
        ranks = hole_cards // 4
        suits = hole_cards % 4
        
        # Simple scoring: high cards + pair bonus + suited bonus
        rank_value = jnp.sum(ranks) * 100
        pair_bonus = jnp.where(ranks[0] == ranks[1], 2000, 0)
        suited_bonus = jnp.where(suits[0] == suits[1], 500, 0)
        
        return rank_value + pair_bonus + suited_bonus
    
    @partial(jax.jit, static_argnums=(0,))
    def _regret_matching(self, regrets: jnp.ndarray) -> jnp.ndarray:
        """
        Convert regrets to strategy using regret matching.
        
        Args:
            regrets: Regret table
            
        Returns:
            Updated strategy table
        """
        # Take positive part of regrets
        positive_regrets = jnp.maximum(regrets, 0.0)
        
        # Sum regrets for each info set
        regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
        
        # Normalize to get probabilities
        strategy = jnp.where(
            regret_sums > self.config.strategy_threshold,
            positive_regrets / regret_sums,
            jnp.ones_like(positive_regrets) / self.config.num_actions
        )
        
        return strategy
    
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

# Factory function for easy creation
def create_trainer(config_path: Optional[str] = None) -> PokerTrainer:
    """
    Create trainer with configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configured PokerTrainer
    """
    if config_path:
        config = TrainerConfig.from_yaml(config_path)
    else:
        config = TrainerConfig()
    
    return PokerTrainer(config)