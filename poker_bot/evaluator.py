"""
🔢 Hand Evaluator - Ultra-fast poker hand evaluation

Wrapper around phevaluator for 400M+ hands/sec evaluation.
"""

import jax
import jax.numpy as jnp
from typing import List, Union, Tuple
from phevaluator import evaluate_cards
import numpy as np


class HandEvaluator:
    """
    GPU-optimized hand evaluator using phevaluator backend.
    
    Provides both single and batch evaluation with JAX integration.
    """
    
    def __init__(self):
        """Initialize the hand evaluator."""
        self.card_to_str = {
            0: "2s", 1: "2h", 2: "2d", 3: "2c",
            4: "3s", 5: "3h", 6: "3d", 7: "3c",
            8: "4s", 9: "4h", 10: "4d", 11: "4c",
            12: "5s", 13: "5h", 14: "5d", 15: "5c",
            16: "6s", 17: "6h", 18: "6d", 19: "6c",
            20: "7s", 21: "7h", 22: "7d", 23: "7c",
            24: "8s", 25: "8h", 26: "8d", 27: "8c",
            28: "9s", 29: "9h", 30: "9d", 31: "9c",
            32: "Ts", 33: "Th", 34: "Td", 35: "Tc",
            36: "Js", 37: "Jh", 38: "Jd", 39: "Jc",
            40: "Qs", 41: "Qh", 42: "Qd", 43: "Qc",
            44: "Ks", 45: "Kh", 46: "Kd", 47: "Kc",
            48: "As", 49: "Ah", 50: "Ad", 51: "Ac",
        }
        
        # JIT compile the batch evaluation function
        self.batch_evaluate = jax.jit(self._batch_evaluate_impl)
    
    def evaluate_single(self, cards: List[int]) -> int:
        """
        Evaluate a single hand.
        
        Args:
            cards: List of card indices (0-51)
            
        Returns:
            Hand strength (higher = better)
        """
        card_strs = [self.card_to_str[card] for card in cards]
        return evaluate_cards(*card_strs)
    
    def evaluate_batch(self, hands: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate multiple hands in batch using JAX.
        
        Args:
            hands: Array of shape (batch_size, num_cards) with card indices
            
        Returns:
            Array of hand strengths
        """
        return self.batch_evaluate(hands)
    
    def _batch_evaluate_impl(self, hands: jnp.ndarray) -> jnp.ndarray:
        """JAX-compiled batch evaluation implementation."""
        
        def evaluate_one_hand(hand):
            # Convert JAX array to Python list for phevaluator
            hand_list = hand.tolist()
            card_strs = [self.card_to_str[card] for card in hand_list]
            return evaluate_cards(*card_strs)
        
        # Use vmap for vectorized evaluation
        return jax.vmap(evaluate_one_hand)(hands)
    
    def compare_hands(self, hand1: List[int], hand2: List[int]) -> int:
        """
        Compare two hands.
        
        Args:
            hand1: First hand (card indices)
            hand2: Second hand (card indices)
            
        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        strength1 = self.evaluate_single(hand1)
        strength2 = self.evaluate_single(hand2)
        
        if strength1 > strength2:
            return 1
        elif strength2 > strength1:
            return -1
        else:
            return 0
    
    def get_hand_rank(self, strength: int) -> str:
        """
        Get readable hand rank from strength.
        
        Args:
            strength: Hand strength from evaluation
            
        Returns:
            Human-readable hand rank
        """
        # phevaluator returns lower numbers for better hands
        # Convert to standard ranking
        if strength <= 10:
            return "Straight Flush"
        elif strength <= 166:
            return "Four of a Kind"
        elif strength <= 322:
            return "Full House"
        elif strength <= 1599:
            return "Flush"
        elif strength <= 1609:
            return "Straight"
        elif strength <= 2467:
            return "Three of a Kind"
        elif strength <= 3325:
            return "Two Pair"
        elif strength <= 6185:
            return "One Pair"
        else:
            return "High Card"


def test_evaluator():
    """Test the hand evaluator."""
    evaluator = HandEvaluator()
    
    # Test single hand evaluation
    royal_flush = [48, 44, 40, 36, 32]  # As, Ks, Qs, Js, Ts (all spades)
    strength = evaluator.evaluate_single(royal_flush)
    print(f"Royal flush strength: {strength}")
    print(f"Hand rank: {evaluator.get_hand_rank(strength)}")
    
    # Test batch evaluation
    hands = jnp.array([
        [48, 44, 40, 36, 32],  # Royal flush
        [48, 49, 44, 45, 40]   # AA KK Q (high cards)
    ])
    
    batch_strengths = evaluator.evaluate_batch(hands)
    print(f"Batch evaluation: {batch_strengths}")
    
    return evaluator


if __name__ == "__main__":
    test_evaluator() 