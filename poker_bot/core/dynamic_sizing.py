import jax
import jax.numpy as jnp

@jax.jit
def optimize_bet_sizing(strategy: jnp.ndarray, hole_cards: jnp.ndarray, 
                       community_cards: jnp.ndarray, pot_size: jnp.ndarray) -> str:
    """
    Convert strategy probabilities to optimal action with dynamic sizing.
    
    Args:
        strategy: [9] adjusted strategy probabilities  
        hole_cards: [2] hole cards
        community_cards: [5] community cards  
        pot_size: current pot size
        
    Returns:
        optimal_action: best action string
    """
    from .starting_hands import classify_starting_hand
    from .board_analysis import analyze_board_texture
    
    hand_strength = classify_starting_hand(hole_cards)
    board_wetness = analyze_board_texture(community_cards)
    
    # Sample action from strategy
    action_idx = jnp.argmax(strategy)
    
    actions = ["FOLD", "CHECK", "CALL", "BET_SMALL", "BET_MED", "BET_LARGE", 
               "RAISE_SMALL", "RAISE_MED", "ALL_IN"]
    
    # Dynamic sizing logic for betting actions
    def optimize_bet_action(base_action):
        # Strong hands on dry boards -> larger bets
        # Bluffs on wet boards -> larger bets  
        # Medium hands -> smaller bets
        
        is_value_bet = hand_strength > 0.6
        is_bluff = hand_strength < 0.3
        
        size_multiplier = jnp.where(
            is_value_bet & (board_wetness < 0.4), 1.5,  # Large value bet on dry board
            jnp.where(
                is_bluff & (board_wetness > 0.6), 1.3,   # Large bluff on wet board
                jnp.where(
                    (hand_strength >= 0.3) & (hand_strength <= 0.6), 0.7, 1.0  # Small bet with medium hands
                )
            )
        )
        
        return jnp.where(
            size_multiplier > 1.2, "BET_LARGE",
            jnp.where(size_multiplier < 0.8, "BET_SMALL", base_action)
        )
    
    base_action = actions[action_idx]
    
    # Apply dynamic sizing to betting actions
    final_action = jnp.where(
        (action_idx >= 3) & (action_idx <= 5),  # BET actions
        optimize_bet_action(base_action),
        jnp.where(
            (action_idx >= 6) & (action_idx <= 7),  # RAISE actions  
            optimize_bet_action(base_action.replace("BET", "RAISE")),
            base_action
        )
    )
    
    return final_action 