import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def optimize_bet_sizing(strategy: jnp.ndarray, hole_cards: jnp.ndarray, 
                       community_cards: jnp.ndarray, pot_size: jnp.ndarray) -> jnp.ndarray:
    """
    Convert strategy probabilities to optimal action index with dynamic sizing.
    
    Args:
        strategy: [9] adjusted strategy probabilities  
        hole_cards: [2] hole cards
        community_cards: [5] community cards  
        pot_size: current pot size
        
    Returns:
        optimal_action_idx: best action index (0-8)
    """
    from .starting_hands import classify_starting_hand
    from .board_analysis import analyze_board_texture
    
    hand_strength = classify_starting_hand(hole_cards)
    board_wetness = analyze_board_texture(community_cards)
    
    # Sample action from strategy
    base_action_idx = jnp.argmax(strategy)
    
    # Dynamic sizing logic for betting actions
    is_value_bet = hand_strength > 0.6
    is_bluff = hand_strength < 0.3
    
    # Size adjustments based on situation
    should_size_up = (is_value_bet & (board_wetness < 0.4)) | (is_bluff & (board_wetness > 0.6))
    should_size_down = (hand_strength >= 0.3) & (hand_strength <= 0.6)
    
    # Action mapping with sizing adjustments
    # 0: FOLD, 1: CHECK, 2: CALL, 3: BET_SMALL, 4: BET_MED, 5: BET_LARGE, 6: RAISE_SMALL, 7: RAISE_MED, 8: ALL_IN
    
    def adjust_action(action_idx):
        # If it's a betting action (3-5), apply sizing
        is_bet_action = (action_idx >= 3) & (action_idx <= 5)
        is_raise_action = (action_idx >= 6) & (action_idx <= 7)
        
        # For betting actions
        adjusted_bet = jnp.where(
            should_size_up, 5,  # BET_LARGE
            jnp.where(should_size_down, 3, action_idx)  # BET_SMALL or keep original
        )
        
        # For raising actions  
        adjusted_raise = jnp.where(
            should_size_up, 7,  # RAISE_MED
            jnp.where(should_size_down, 6, action_idx)  # RAISE_SMALL or keep original
        )
        
        # Return adjusted action
        return jnp.where(
            is_bet_action, adjusted_bet,
            jnp.where(is_raise_action, adjusted_raise, action_idx)
        )
    
    final_action_idx = adjust_action(base_action_idx)
    return jnp.clip(final_action_idx, 0, 8).astype(jnp.int32)

def convert_action_idx_to_string(action_idx: int) -> str:
    """Convert action index to string (NON-JIT function)."""
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", 
               "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    return actions[int(action_idx)] 