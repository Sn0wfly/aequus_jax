# poker_bot/core/__init__.py

"""
Core training and game engine components for Aequus JAX
"""

from .trainer import PokerTrainer, TrainerConfig, create_trainer, load_hand_evaluation_lut
from .bucketing import compute_info_set_id, validate_bucketing_system  
from .validation import PokerAIValidator, quick_validation, detailed_validation
from .full_game_engine import (
    GameState, play_one_game, batch_play, 
    unified_batch_simulation_with_lut, initial_state_for_idx
)

__all__ = [
    # Trainer components
    "PokerTrainer", "TrainerConfig", "create_trainer", "load_hand_evaluation_lut",
    
    # Bucketing system
    "compute_info_set_id", "validate_bucketing_system",
    
    # Validation system
    "PokerAIValidator", "quick_validation", "detailed_validation",
    
    # Game engine
    "GameState", "play_one_game", "batch_play",
    "unified_batch_simulation_with_lut", "initial_state_for_idx"
]