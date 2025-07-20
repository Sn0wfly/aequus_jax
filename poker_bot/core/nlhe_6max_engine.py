"""
True NLHE 6-Max Game Engine with 9-action system.
Implements proper NLHE rules with pot-relative betting and position play.
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from jax.tree_util import register_pytree_node_class
from typing import Tuple

# Constants for NLHE 6-Max
MAX_PLAYERS = 6
STARTING_STACK = 1000  # 100bb = 1000 chips (1bb = 10)
SMALL_BLIND = 5
BIG_BLIND = 10
ANTE = 0

# 9-action NLHE system
class Action(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_0_5X = 3    # 50% pot
    BET_0_75X = 4   # 75% pot
    BET_1X = 5      # 100% pot
    BET_1_5X = 6    # 150% pot
    BET_2X = 7      # 200% pot
    ALL_IN = 8      # All remaining chips

# Street progression
class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

# Position mapping for 6-max
POSITIONS = ["BTN", "SB", "BB", "UTG", "MP", "CO"]

@register_pytree_node_class
@dataclass
class NLHEState:
    """Complete NLHE 6-max game state."""
    stacks: jax.Array          # [6] - Player stacks in chips
    bets: jax.Array            # [6] - Current bets in chips
    player_status: jax.Array   # [6] - 0=active, 1=folded, 2=all-in
    hole_cards: jax.Array      # [6, 2] - Player hole cards
    community_cards: jax.Array # [5] - Community cards
    current_player: jax.Array  # [1] - Index of current player
    street: jax.Array          # [1] - Current street
    pot: jax.Array             # [1] - Current pot size
    dealer: jax.Array          # [1] - Dealer position
    round_bets: jax.Array      # [6] - Bets this round
    acted_this_round: jax.Array # [6] - Players who acted this round
    
    def tree_flatten(self):
        return (
            self.stacks, self.bets, self.player_status, self.hole_cards,
            self.community_cards, self.current_player, self.street, self.pot,
            self.dealer, self.round_bets, self.acted_this_round
        ), None
    
    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

# Action utilities
@jax.jit
def calculate_bet_size(action: int, pot_size: int, stack_size: int, to_call: int) -> int:
    """Calculate exact bet size for given action."""
    bet_ratios = jnp.array([0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 1.0])
    ratio = bet_ratios[action]
    
    calculated_bet = jnp.round(pot_size * ratio).astype(jnp.int32)
    is_all_in = (action == Action.ALL_IN)
    
    return jnp.where(is_all_in, stack_size, jnp.minimum(calculated_bet, stack_size))

@jax.jit
def get_legal_actions(state: NLHEState) -> jnp.ndarray:
    """Get legal actions for current player."""
    actions = jnp.zeros(9, dtype=jnp.bool_)
    player = state.current_player[0]
    
    can_act = (state.player_status[player] == 0)
    stack = state.stacks[player]
    current_bet = state.bets[player]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current_bet
    
    # Fold is always legal if can act
    actions = actions.at[Action.FOLD].set(can_act & (to_call > 0))
    
    # Check is legal if no bet to call
    actions = actions.at[Action.CHECK].set(can_act & (to_call == 0))
    
    # Call is legal if there's a bet
    actions = actions.at[Action.CALL].set(can_act & (to_call > 0))
    
    # Betting actions are legal if player has chips
    can_bet = can_act & (stack > 0)
    min_bet = jnp.maximum(1, BIG_BLIND)
    
    # Calculate bet amounts for each betting action
    pot_size = state.pot[0]
    for action_idx in [Action.BET_0_5X, Action.BET_0_75X, Action.BET_1X, 
                       Action.BET_1_5X, Action.BET_2X]:
        bet_amount = calculate_bet_size(action_idx, pot_size, stack, to_call)
        is_valid = can_bet & (bet_amount >= min_bet)
        actions = actions.at[action_idx].set(is_valid)
    
    # All-in is legal if player has chips
    actions = actions.at[Action.ALL_IN].set(can_act & (stack > 0))
    
    return actions

@jax.jit
def get_next_active_player(status: jax.Array, start: int) -> int:
    """Get next active player."""
    for i in range(MAX_PLAYERS):
        idx = (start + i) % MAX_PLAYERS
        if status[idx] == 0:  # Active player
            return idx
    return start

@jax.jit
def is_betting_round_complete(state: NLHEState) -> bool:
    """Check if betting round is complete."""
    active_players = (state.player_status == 0)
    num_active = jnp.sum(active_players)
    
    max_bet = jnp.max(state.bets)
    all_called = jnp.all((state.bets == max_bet) | (state.player_status != 0))
    all_acted = jnp.all(state.acted_this_round | (state.player_status != 0))
    
    return (num_active <= 1) | (all_called & all_acted)

@jax.jit
def initialize_game(key: jax.random.PRNGKey) -> NLHEState:
    """Initialize new NLHE 6-max game."""
    key, subkey = jax.random.split(key)
    
    # Deal hole cards to all players
    hole_cards = jax.random.permutation(subkey, jnp.arange(52)).reshape(MAX_PLAYERS, 2)
    
    # Initialize stacks
    stacks = jnp.full(MAX_PLAYERS, STARTING_STACK)
    
    # Post blinds
    stacks = stacks.at[0].add(-SMALL_BLIND).at[1].add(-BIG_BLIND)
    
    # Initialize bets
    bets = jnp.zeros(MAX_PLAYERS)
    bets = bets.at[0].set(SMALL_BLIND).at[1].set(BIG_BLIND)
    
    return NLHEState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros(MAX_PLAYERS),
        hole_cards=hole_cards,
        community_cards=jnp.full(5, -1),
        current_player=jnp.array([2]),
        street=jnp.array([0]),
        pot=jnp.array([15]),
        dealer=jnp.array([0]),
        round_bets=jnp.zeros(MAX_PLAYERS),
        acted_this_round=jnp.zeros(MAX_PLAYERS)
    )

@jax.jit
def play_one_game(key: jax.random.PRNGKey) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Play one complete NLHE 6-max game."""
    key, subkey = jax.random.split(key)
    
    # Initialize game
    state = initialize_game(key)
    
    # Play through all streets
    state = play_street(state, 3)  # Flop
    state = play_street(state, 1)  # Turn
    state = play_street(state, 1)  # River
    
    # Resolve showdown
    payoffs = resolve_showdown(state)
    
    return payoffs, state.hole_cards, state.community_cards

@jax.jit
def play_street(state: NLHEState, num_cards: int) -> NLHEState:
    """Play through a street (flop, turn, river)."""
    # Deal community cards
    state = deal_street(state, num_cards)
    state = run_betting_round(state)
    return state

@jax.jit
def deal_street(state: NLHEState, num_cards: int) -> NLHEState:
    """Deal community cards for a street."""
    start = state.deck_ptr[0]
    cards = lax.dynamic_slice(state.deck, (start,), (num_cards,))
    comm = lax.dynamic_update_slice(state.community_cards, cards, (start,))
    return replace(
        state,
        comm_cards=comm,
        deck_ptr=state.deck_ptr + num_cards,
        acted_this_round=jnp.zeros(MAX_PLAYERS),
        cur_player=jnp.array([0])
    )

@jax.jit
def run_betting_round(state: NLHEState) -> NLHEState:
    """Run betting round until completion."""
    cond = lambda s: ~is_betting_round_complete(s)
    return lax.while_loop(cond, lambda s: s, state)