# poker_bot/__init__.py

"""
Aequus JAX: High-Performance Poker AI
Pure JAX implementation with clean architecture
"""

__version__ = "2.0.0"
__author__ = "Aequus Team"

# Core training components
from .core.trainer import PokerTrainer, TrainerConfig, create_trainer
from .core.bucketing import compute_info_set_id, validate_bucketing_system
from .core.validation import PokerAIValidator, quick_validation, detailed_validation

# Bot and evaluation
from .bot import PokerBot
from .evaluator import HandEvaluator

# System utilities
from .gpu_config import get_device_info, init_gpu_environment
from .memory import get_memory_usage, MemoryMonitor

__all__ = [
    # Training system
    "PokerTrainer", "TrainerConfig", "create_trainer",
    
    # Bucketing and validation
    "compute_info_set_id", "validate_bucketing_system",
    "PokerAIValidator", "quick_validation", "detailed_validation",
    
    # Bot components
    "PokerBot", "HandEvaluator",
    
    # System utilities
    "get_device_info", "init_gpu_environment",
    "get_memory_usage", "MemoryMonitor"
]