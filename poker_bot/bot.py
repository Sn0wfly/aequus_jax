# poker_bot/bot.py

"""
Poker AI Agent (Playable Bot)

This module contains the PokerBot class, which loads a trained GTO model
and uses it to make real-time game decisions.
"""

import numpy as np
import pickle
import logging
import jax.numpy as jnp
from typing import Dict, Any

# Import from the clean core structure
from .core.trainer import TrainerConfig 
from .core.bucketing import compute_info_set_id

logger = logging.getLogger(__name__)

class PokerBot:
    """
    An AI agent that plays poker using a pre-trained GTO strategy.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the bot by loading the trained model.

        Args:
            model_path: Path to the GTO model .pkl file.
        """
        self.model_path = model_path
        self.regrets: np.ndarray = None
        self.strategy: np.ndarray = None
        self.config: TrainerConfig = None
        self.iteration = 0

        logger.info(f"🤖 Loading GTO model from {model_path}...")
        self._load_model()

    def _load_model(self):
        """Load the GTO model data from .pkl file."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract main model components
            self.regrets = model_data.get('regrets', np.array([]))
            self.strategy = model_data.get('strategy', np.array([]))
            self.iteration = model_data.get('iteration', 0)
            
            # Load configuration
            config_data = model_data.get('config', {})
            if isinstance(config_data, TrainerConfig):
                self.config = config_data
            elif isinstance(config_data, dict):
                self.config = TrainerConfig(**config_data)
            else:
                self.config = TrainerConfig()  # Default config

            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Iteration: {self.iteration}")
            logger.info(f"   Strategy shape: {self.strategy.shape}")
            logger.info(f"   Regrets shape: {self.regrets.shape}")
            
        except FileNotFoundError:
            logger.error(f"❌ Error: Model file not found at '{self.model_path}'")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def get_action(self, game_state: dict) -> str:
        """
        Main bot function. Given a game state, returns the best action.
        
        Args:
            game_state: Dictionary containing:
                - player_id: Current player index
                - hole_cards: Player's hole cards [card1, card2]
                - community_cards: Community cards [c1, c2, c3, c4, c5]
                - pot_size: Current pot size
                - position: Player position (0-5)
                
        Returns:
            Best action as string: "FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"
        """
        try:
            # Extract data directly for bucketing system
            player_id = game_state.get('player_id', 0)
            
            # Get hole cards
            hole_cards = game_state.get('hole_cards', [0, 1])
            if isinstance(hole_cards, list):
                hole_cards = jnp.array(hole_cards, dtype=jnp.int8)
            
            # Get community cards
            comm_cards = game_state.get('community_cards', [-1, -1, -1, -1, -1])
            if isinstance(comm_cards, list):
                comm_cards = jnp.array(comm_cards, dtype=jnp.int8)
            
            # Ensure community cards are correct length
            if len(comm_cards) < 5:
                comm_cards = jnp.pad(comm_cards, (0, 5 - len(comm_cards)), 
                                   constant_values=-1)
            elif len(comm_cards) > 5:
                comm_cards = comm_cards[:5]
            
            # Get pot size
            pot_size = jnp.array([game_state.get('pot_size', 50.0)])
            
            # Get info set using new bucketing system
            info_set_idx = compute_info_set_id(hole_cards, comm_cards, player_id, pot_size)
            
            # Convert to Python int
            info_set_idx = int(info_set_idx)
            
            # Check if info set is valid and within strategy bounds
            if (info_set_idx >= 0 and 
                info_set_idx < self.strategy.shape[0] and 
                self.strategy.shape[1] >= 6):
                
                # Get strategy for this info set
                strategy_probs = self.strategy[info_set_idx]
                actions = ["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"]
                
                # Asegurar que las probabilidades sean válidas para el muestreo
                strategy_probs = np.maximum(strategy_probs, 0)  # Prevenir probabilidades negativas
                prob_sum = np.sum(strategy_probs)
                
                if prob_sum > 1e-6:  # Usar un umbral pequeño para estabilidad
                    strategy_probs = strategy_probs / prob_sum
                else:
                    # Si la suma es cero (un caso raro), fallback a una estrategia uniforme
                    num_actions = len(actions)
                    strategy_probs = np.ones(num_actions) / num_actions
                
                # Asegurar arrays mismo tamaño
                actions = actions[:len(strategy_probs)]
                selected_action = np.random.choice(actions, p=strategy_probs)
                
                logger.debug(f"Info set {info_set_idx}: Strategy={np.round(strategy_probs, 2)} -> Sampled Action: {selected_action}")
                return selected_action
                
            else:
                logger.warning(f"Invalid info set {info_set_idx}, using default action")
                return "CHECK"
                
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            return "CHECK"  # Safe default

# _convert_to_mock_game_state removed - no longer needed with direct array passing

    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the loaded strategy.
        
        Returns:
            Dictionary with strategy statistics
        """
        if self.strategy is None or self.strategy.size == 0:
            return {"error": "No strategy loaded"}
        
        return {
            "strategy_shape": self.strategy.shape,
            "regrets_shape": self.regrets.shape if self.regrets is not None else None,
            "iteration": self.iteration,
            "strategy_mean": float(np.mean(self.strategy)),
            "strategy_std": float(np.std(self.strategy)),
            "strategy_entropy": self._compute_average_entropy(),
            "model_path": self.model_path
        }
    
    def _compute_average_entropy(self) -> float:
        """Compute average entropy of the strategy"""
        if self.strategy is None:
            return 0.0
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        strategy_safe = np.clip(self.strategy, eps, 1.0)
        
        # Compute entropy for each info set
        log_probs = np.log(strategy_safe)
        entropy_per_info_set = -np.sum(self.strategy * log_probs, axis=1)
        
        return float(np.mean(entropy_per_info_set))

    def evaluate_hand_strength(self, game_state: dict) -> float:
        """
        Evaluate the relative strength of current hand.
        
        Args:
            game_state: Current game state
            
        Returns:
            Hand strength estimate (0.0 to 1.0)
        """
        try:
            hole_cards = game_state.get('hole_cards', [0, 1])
            if isinstance(hole_cards, list):
                hole_cards = np.array(hole_cards)
            
            # Simple hand strength estimation
            ranks = hole_cards // 4
            suits = hole_cards % 4
            
            # Base strength from high cards
            rank_strength = np.sum(ranks) / 24.0  # Max sum is 24 (AA)
            
            # Pair bonus
            pair_bonus = 0.3 if ranks[0] == ranks[1] else 0.0
            
            # Suited bonus
            suited_bonus = 0.1 if suits[0] == suits[1] else 0.0
            
            # High card bonus (A, K, Q)
            high_cards = np.sum(ranks >= 10) / 2.0 * 0.2
            
            strength = rank_strength + pair_bonus + suited_bonus + high_cards
            return min(1.0, strength)
            
        except Exception as e:
            logger.error(f"Error evaluating hand strength: {e}")
            return 0.5  # Neutral strength on error

# Utility function for testing
def test_bot_loading(model_path: str) -> bool:
    """
    Test if a bot can be loaded successfully.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if loading successful
    """
    try:
        bot = PokerBot(model_path)
        
        # Test basic functionality
        test_state = {
            'player_id': 0,
            'hole_cards': [48, 44],  # AA
            'community_cards': [10, 11, 12, -1, -1],
            'pot_size': 100.0,
            'position': 2
        }
        
        action = bot.get_action(test_state)
        logger.info(f"✅ Bot test successful: {action}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bot test failed: {e}")
        return False