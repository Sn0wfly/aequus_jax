# poker_bot/__init__.py

"""
Main package exports for Aequus JAX Poker Bot
"""

from .core.trainer import PokerTrainer, TrainerConfig, create_trainer
from .core.bucketing import compute_info_set_id, validate_bucketing_system  
from .core.validation import PokerAIValidator, quick_validation, detailed_validation
from .core.full_game_engine import (
    GameState, 
    play_one_game, 
    batch_play,
    initial_state_for_idx
)

# Import the bot from the main level
from .bot import PokerBot

__all__ = [
    # Trainer components
    "PokerTrainer", "TrainerConfig", "create_trainer",
    
    # Bucketing system
    "compute_info_set_id", "validate_bucketing_system",
    
    # Validation system
    "PokerAIValidator", "quick_validation", "detailed_validation",
    
    # Game engine
    "GameState", "play_one_game", "batch_play", "initial_state_for_idx",
    
    # Bot
    "PokerBot"
]