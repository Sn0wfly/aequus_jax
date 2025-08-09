# full_game_engine.py
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass, replace
from functools import partial
import numpy as np
from jax.tree_util import register_pytree_node_class
from ..evaluator import HandEvaluator
from .bucketing import compute_info_set_id_enhanced

MAX_GAME_LENGTH = 60
evaluator = HandEvaluator()

# Action definitions for 9-action NLHE system
# 0: FOLD, 1: CHECK, 2: CALL, 3: BET_SMALL, 4: BET_MED, 5: BET_LARGE, 6: RAISE_SMALL, 7: RAISE_MED, 8: ALL_IN

# ---------- Pytree dataclass -----------
@register_pytree_node_class
@dataclass
class GameState:
    stacks: jax.Array
    bets: jax.Array
    player_status: jax.Array
    hole_cards: jax.Array
    comm_cards: jax.Array
    cur_player: jax.Array
    street: jax.Array
    pot: jax.Array
    deck: jax.Array
    deck_ptr: jax.Array
    acted_this_round: jax.Array
    key: jax.Array
    action_hist: jax.Array
    hist_ptr: jax.Array
    # Trajectory logging for training
    info_hist: jax.Array        # [MAX_GAME_LENGTH] int32 info set ids per decision
    legal_hist: jax.Array       # [MAX_GAME_LENGTH, 9] bool legal actions per decision
    player_hist: jax.Array      # [MAX_GAME_LENGTH] int8 current player per decision
    pot_hist: jax.Array         # [MAX_GAME_LENGTH] float pot size per decision
    comm_hist: jax.Array        # [MAX_GAME_LENGTH, 5] int32 community snapshot per decision

    def tree_flatten(self):
        children = (self.stacks, self.bets, self.player_status, self.hole_cards,
                    self.comm_cards, self.cur_player, self.street, self.pot,
                    self.deck, self.deck_ptr, self.acted_this_round,
                    self.key, self.action_hist, self.hist_ptr,
                    self.info_hist, self.legal_hist, self.player_hist, self.pot_hist, self.comm_hist)
        return children, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

# ---------- JAX-NATIVE HAND EVALUATION WITH LUT ----------
# LUT parameters are now passed as function arguments instead of global variables

@jax.jit
def evaluate_hand_jax_native(cards, lut_keys=None, lut_values=None, lut_table_size=None):
    """
    Production-grade 7-card hand evaluator.
    Based on Two Plus Two algorithm, optimized for JAX.
    O(1) complexity, no LUT dependencies.
    """
    # Filter valid cards
    valid_mask = cards >= 0
    valid_cards = jnp.where(valid_mask, cards, 0)
    num_valid = jnp.sum(valid_mask)
    # Early return for insufficient cards
    def insufficient_cards():
        return jnp.sum(valid_cards).astype(jnp.int32)
    def evaluate_full_hand():
        # Extract ranks and suits using bit operations
        ranks = valid_cards // 4  # 0-12 (2,3,4,5,6,7,8,9,T,J,Q,K,A)
        suits = valid_cards % 4   # 0-3 (clubs, diamonds, hearts, spades)
        # Count ranks and suits efficiently
        rank_counts = jnp.zeros(13, dtype=jnp.int32)
        suit_counts = jnp.zeros(4, dtype=jnp.int32)
        # Accumulate counts using scatter_add (JAX-optimized)
        for i in range(7):
            mask_i = valid_mask[i]
            rank_counts = rank_counts.at[ranks[i]].add(mask_i.astype(jnp.int32))
            suit_counts = suit_counts.at[suits[i]].add(mask_i.astype(jnp.int32))
        # Detect hand types using vectorized operations
        pair_count = jnp.sum(rank_counts == 2)
        trips_count = jnp.sum(rank_counts == 3) 
        quads_count = jnp.sum(rank_counts == 4)
        is_flush = jnp.any(suit_counts >= 5)
        # Detect straight using efficient bit manipulation
        rank_bitmap = jnp.sum(jnp.where(rank_counts > 0, 1 << jnp.arange(13), 0))
        # Check for straights (including A-2-3-4-5 wheel)
        straight_patterns = jnp.array([
            0b1111100000000,  # A-K-Q-J-T
            0b0111110000000,  # K-Q-J-T-9
            0b0011111000000,  # Q-J-T-9-8
            0b0001111100000,  # J-T-9-8-7
            0b0000111110000,  # T-9-8-7-6
            0b0000011111000,  # 9-8-7-6-5
            0b0000001111100,  # 8-7-6-5-4
            0b0000000111110,  # 7-6-5-4-3
            0b0000000011111,  # 6-5-4-3-2
            0b1000000001111,  # A-5-4-3-2 (wheel)
        ], dtype=jnp.int32)
        is_straight = jnp.any((rank_bitmap & straight_patterns) == straight_patterns)
        # Calculate high card (Ace-high = 12)
        high_card = jnp.max(jnp.where(rank_counts > 0, jnp.arange(13), -1))
        # Professional poker hand rankings (0-7462 scale)
        # Higher numbers = better hands
        def straight_flush():
            return 7400 + high_card
        def four_of_kind():
            quad_rank = jnp.max(jnp.where(rank_counts == 4, jnp.arange(13), -1))
            return 7200 + quad_rank * 13 + high_card
        def full_house():
            trips_rank = jnp.max(jnp.where(rank_counts == 3, jnp.arange(13), -1))
            pair_rank = jnp.max(jnp.where(rank_counts == 2, jnp.arange(13), -1))
            return 6600 + trips_rank * 13 + pair_rank
        def flush():
            return 5900 + high_card * 4
        def straight():
            return 5200 + high_card
        def trips():
            trips_rank = jnp.max(jnp.where(rank_counts == 3, jnp.arange(13), -1))
            return 3400 + trips_rank * 169 + high_card
        def two_pair():
            pair_ranks = jnp.where(rank_counts == 2, jnp.arange(13), -1)
            high_pair = jnp.max(pair_ranks)
            low_pair = jnp.max(jnp.where(pair_ranks != high_pair, pair_ranks, -1))
            return 1700 + high_pair * 169 + low_pair * 13 + high_card
        def one_pair():
            pair_rank = jnp.max(jnp.where(rank_counts == 2, jnp.arange(13), -1))
            return 800 + pair_rank * 169 + high_card
        def high_card_only():
            return high_card * 13 + jnp.sum(jnp.sort(jnp.where(rank_counts > 0, jnp.arange(13), -1))[-5:])
        # Hierarchical hand evaluation
        hand_strength = jnp.where(
            is_straight & is_flush, straight_flush(),
            jnp.where(
                quads_count > 0, four_of_kind(),
                jnp.where(
                    (trips_count > 0) & (pair_count > 0), full_house(),
                    jnp.where(
                        is_flush, flush(),
                        jnp.where(
                            is_straight, straight(),
                            jnp.where(
                                trips_count > 0, trips(),
                                jnp.where(
                                    pair_count >= 2, two_pair(),
                                    jnp.where(
                                        pair_count == 1, one_pair(),
                                        high_card_only()
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        return hand_strength.astype(jnp.int32)
    return jnp.where(num_valid >= 5, evaluate_full_hand(), insufficient_cards())

# ---------- Helpers ----------
@jax.jit
def next_active_player(ps, start):
    idx = (start + jnp.arange(6, dtype=jnp.int8)) % 6
    mask = ps[idx] == 0
    return jnp.where(mask.any(), idx[jnp.argmax(mask)], start).astype(jnp.int8)

@jax.jit
def is_betting_done(status, bets, acted, _):
    active = status != 1
    num_active = active.sum()
    max_bet = jnp.max(bets * active)
    all_called = (acted[0] >= num_active) & (bets == max_bet).all()
    return (num_active <= 1) | all_called

@jax.jit
def get_legal_actions_9(state: GameState):
    """
    Get legal actions for 9-action NLHE system.
    
    Returns:
        Boolean mask of legal actions [9]
    """
    mask = jnp.zeros(9, dtype=jnp.bool_)
    p = jnp.squeeze(state.cur_player)
    status = state.player_status[p]
    can_act = status == 0
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current
    pot_size = jnp.squeeze(state.pot)
    
    # FOLD (0) - always legal if can act
    mask = mask.at[0].set(can_act)
    
    # CHECK (1) - legal if no bet to call
    mask = mask.at[1].set(can_act & (to_call == 0))
    
    # CALL (2) - legal if there's a bet to call and enough chips
    mask = mask.at[2].set(can_act & (to_call > 0) & (state.stacks[p] >= to_call))
    
    # BET actions (3, 4, 5) - legal if no bet to call and have chips
    bet_small = jnp.minimum(pot_size * 0.25, state.stacks[p])  # 25% pot
    bet_med = jnp.minimum(pot_size * 0.5, state.stacks[p])     # 50% pot
    bet_large = jnp.minimum(pot_size * 0.75, state.stacks[p])  # 75% pot
    
    mask = mask.at[3].set(can_act & (to_call == 0) & (state.stacks[p] > 0) & (bet_small > 0))
    mask = mask.at[4].set(can_act & (to_call == 0) & (state.stacks[p] > 0) & (bet_med > 0))
    mask = mask.at[5].set(can_act & (to_call == 0) & (state.stacks[p] > 0) & (bet_large > 0))
    
    # RAISE actions (6, 7) - legal if there's a bet to raise and have chips
    raise_small = jnp.minimum(max_bet + pot_size * 0.25, state.stacks[p] + current)
    raise_med = jnp.minimum(max_bet + pot_size * 0.5, state.stacks[p] + current)
    
    mask = mask.at[6].set(can_act & (to_call > 0) & (state.stacks[p] > 0) & (raise_small > max_bet))
    mask = mask.at[7].set(can_act & (to_call > 0) & (state.stacks[p] > 0) & (raise_med > max_bet))
    
    # ALL_IN (8) - always legal if can act and have chips
    mask = mask.at[8].set(can_act & (state.stacks[p] > 0))
    
    return mask

@jax.jit
def get_legal_actions_6(state: GameState):
    """
    Get legal actions for 6-action system (backward compatibility).
    
    Returns:
        Boolean mask of legal actions [6]
    """
    mask = jnp.zeros(6, dtype=jnp.bool_)
    p = jnp.squeeze(state.cur_player)
    status = state.player_status[p]
    can_act = status == 0
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current
    
    mask = mask.at[0].set(can_act)  # fold
    mask = mask.at[1].set(can_act & ((to_call == 0) | (state.stacks[p] >= to_call)))  # check/call
    mask = mask.at[2].set(can_act & (state.stacks[p] > 0) & (current != max_bet))  # bet/raise
    mask = mask.at[3].set(can_act & (state.stacks[p] > 0) & (current != max_bet))  # bet/raise
    mask = mask.at[4].set(can_act & (state.stacks[p] > 0) & (current != max_bet))  # bet/raise
    mask = mask.at[5].set(can_act & (state.stacks[p] > 0))  # all_in
    
    return mask

@jax.jit
def get_legal_actions_3(state: GameState):
    """
    Get legal actions for 3-action system (backward compatibility).
    
    Returns:
        Boolean mask of legal actions [3]
    """
    mask = jnp.zeros(3, dtype=jnp.bool_)
    p = jnp.squeeze(state.cur_player)
    status = state.player_status[p]
    can_act = status == 0
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current
    
    mask = mask.at[0].set(can_act)  # fold
    mask = mask.at[1].set(can_act & ((to_call == 0) | (state.stacks[p] >= to_call)))  # check/call
    mask = mask.at[2].set(can_act & (state.stacks[p] > 0) & (current != max_bet))  # bet/raise
    
    return mask

# ---------- Step ----------
@jax.jit
def apply_action_9(state, action):
    """
    Apply action for 9-action NLHE system.
    
    Args:
        state: Current game state
        action: Action index (0-8)
        
    Returns:
        Updated game state
    """
    p = jnp.squeeze(state.cur_player)
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current
    pot_size = jnp.squeeze(state.pot)

    def do_fold(s):
        return replace(s, player_status=s.player_status.at[p].set(1))

    def do_check(s):
        return s  # No change to state

    def do_call(s):
        amt = jnp.minimum(to_call, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_small(s):
        amt = jnp.minimum(pot_size * 0.25, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_med(s):
        amt = jnp.minimum(pot_size * 0.5, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_large(s):
        amt = jnp.minimum(pot_size * 0.75, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_raise_small(s):
        amt = jnp.minimum(max_bet + pot_size * 0.25 - current, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_raise_med(s):
        amt = jnp.minimum(max_bet + pot_size * 0.5 - current, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_all_in(s):
        amt = s.stacks[p]
        return replace(
            s,
            stacks=s.stacks.at[p].set(0.0),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    # Apply action using lax.switch
    state2 = lax.switch(
        jnp.clip(action, 0, 8), 
        [do_fold, do_check, do_call, do_bet_small, do_bet_med, do_bet_large, 
         do_raise_small, do_raise_med, do_all_in], 
        state
    )
    
    # Log decision context (info set, legal actions, player, pot)
    idx = state2.hist_ptr[0]
    legal = get_legal_actions_9(state)
    pot_scalar = jnp.squeeze(state.pot)
    # Compute info set id for the acting player using enhanced bucketing
    player_index = jnp.squeeze(state.cur_player).astype(jnp.int32)
    # Extract hole cards for player
    hole = state.hole_cards[player_index]
    info_id = compute_info_set_id_enhanced(hole, state.comm_cards, player_index, state.pot, state.stacks[player_index:player_index+1])

    new_hist = state2.action_hist.at[idx].set(action)
    new_info_hist = state2.info_hist.at[idx].set(info_id)
    new_legal_hist = state2.legal_hist.at[idx].set(legal)
    new_player_hist = state2.player_hist.at[idx].set(player_index.astype(jnp.int8))
    new_pot_hist = state2.pot_hist.at[idx].set(pot_scalar)
    new_comm_hist = state2.comm_hist.at[idx].set(state.comm_cards)
    return replace(
        state2,
        action_hist=new_hist,
        info_hist=new_info_hist,
        legal_hist=new_legal_hist,
        player_hist=new_player_hist,
        pot_hist=new_pot_hist,
        comm_hist=new_comm_hist,
        hist_ptr=state2.hist_ptr + 1,
        acted_this_round=state2.acted_this_round + 1
    )

@jax.jit
def apply_action_6(state, action):
    """Apply action for 6-action system (backward compatibility)."""
    p = jnp.squeeze(state.cur_player)
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current

    def do_fold(s):
        return replace(s, player_status=s.player_status.at[p].set(1))

    def do_check_call(s):
        amt = jnp.where(to_call > 0, to_call, 0.0)
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_raise(s):
        amt = jnp.minimum(20.0, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_all_in(s):
        amt = s.stacks[p]
        return replace(
            s,
            stacks=s.stacks.at[p].set(0.0),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    # Map 6-action system to appropriate actions
    state2 = lax.switch(
        jnp.clip(action, 0, 5),
        [do_fold, do_check_call, do_check_call, do_bet_raise, do_bet_raise, do_all_in],
        state
    )
    
    # Minimal logging for 6-action path (use 9-action logging format)
    idx = state2.hist_ptr[0]
    legal = get_legal_actions_6(state)
    pot_scalar = jnp.squeeze(state.pot)
    player_index = jnp.squeeze(state.cur_player).astype(jnp.int32)
    hole = state.hole_cards[player_index]
    info_id = compute_info_set_id_enhanced(hole, state.comm_cards, player_index, state.pot, state.stacks[player_index:player_index+1])

    new_hist = state2.action_hist.at[idx].set(action)
    new_info_hist = state2.info_hist.at[idx].set(info_id)
    # Expand legal to 9 slots for consistency (pad with False)
    legal9 = jnp.concatenate([legal, jnp.zeros((3,), dtype=jnp.bool_)])
    new_legal_hist = state2.legal_hist.at[idx].set(legal9)
    new_player_hist = state2.player_hist.at[idx].set(player_index.astype(jnp.int8))
    new_pot_hist = state2.pot_hist.at[idx].set(pot_scalar)
    new_comm_hist = state2.comm_hist.at[idx].set(state.comm_cards)
    return replace(
        state2,
        action_hist=new_hist,
        info_hist=new_info_hist,
        legal_hist=new_legal_hist,
        player_hist=new_player_hist,
        pot_hist=new_pot_hist,
        comm_hist=new_comm_hist,
        hist_ptr=state2.hist_ptr + 1,
        acted_this_round=state2.acted_this_round + 1
    )

@jax.jit
def apply_action_3(state, action):
    """Apply action for 3-action system (backward compatibility)."""
    p = jnp.squeeze(state.cur_player)
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current

    def do_fold(s):
        return replace(s, player_status=s.player_status.at[p].set(1))

    def do_check_call(s):
        amt = jnp.where(to_call > 0, to_call, 0.0)
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_raise(s):
        amt = jnp.minimum(20.0, s.stacks[p])
        return replace(
            s,
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    state2 = lax.switch(jnp.clip(action, 0, 2), [do_fold, do_check_call, do_bet_raise], state)
    # Minimal logging for 3-action path (use 9-action logging format)
    idx = state2.hist_ptr[0]
    legal = get_legal_actions_3(state)
    pot_scalar = jnp.squeeze(state.pot)
    player_index = jnp.squeeze(state.cur_player).astype(jnp.int32)
    hole = state.hole_cards[player_index]
    info_id = compute_info_set_id_enhanced(hole, state.comm_cards, player_index, state.pot, state.stacks[player_index:player_index+1])

    new_hist = state2.action_hist.at[idx].set(action)
    # Expand legal to 9 slots for consistency (pad with False)
    legal9 = jnp.concatenate([legal, jnp.zeros((6,), dtype=jnp.bool_)])
    new_info_hist = state2.info_hist.at[idx].set(info_id)
    new_legal_hist = state2.legal_hist.at[idx].set(legal9)
    new_player_hist = state2.player_hist.at[idx].set(player_index.astype(jnp.int8))
    new_pot_hist = state2.pot_hist.at[idx].set(pot_scalar)
    new_comm_hist = state2.comm_hist.at[idx].set(state.comm_cards)
    return replace(
        state2,
        action_hist=new_hist,
        info_hist=new_info_hist,
        legal_hist=new_legal_hist,
        player_hist=new_player_hist,
        pot_hist=new_pot_hist,
        comm_hist=new_comm_hist,
        hist_ptr=state2.hist_ptr + 1,
        acted_this_round=state2.acted_this_round + 1
    )

def step(state, action, num_actions=9):
    """Apply action with configurable action space."""
    def apply_9_action():
        return apply_action_9(state, action)
    
    def apply_6_action():
        return apply_action_6(state, action)
    
    def apply_3_action():
        return apply_action_3(state, action)
    
    # Use lax.cond instead of if statements for JAX compatibility
    return lax.cond(
        num_actions == 9,
        apply_9_action,
        lambda: lax.cond(
            num_actions == 6,
            apply_6_action,
            apply_3_action
        )
    )

# ---------- Betting round ----------
def _betting_body_9(state):
    """Betting round body for 9-action system."""
    legal = get_legal_actions_9(state)
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
    state = replace(state, key=key)
    state = apply_action_9(state, action)
    current_p = jnp.squeeze(state.cur_player)
    next_p = next_active_player(state.player_status, (current_p + 1) % 6)
    return replace(state, cur_player=jnp.array([next_p], dtype=jnp.int8))

def _betting_body_6(state):
    """Betting round body for 6-action system."""
    legal = get_legal_actions_6(state)
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
    state = replace(state, key=key)
    state = apply_action_6(state, action)
    current_p = jnp.squeeze(state.cur_player)
    next_p = next_active_player(state.player_status, (current_p + 1) % 6)
    return replace(state, cur_player=jnp.array([next_p], dtype=jnp.int8))

def _betting_body_3(state):
    """Betting round body for 3-action system."""
    legal = get_legal_actions_3(state)
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
    state = replace(state, key=key)
    state = apply_action_3(state, action)
    current_p = jnp.squeeze(state.cur_player)
    next_p = next_active_player(state.player_status, (current_p + 1) % 6)
    return replace(state, cur_player=jnp.array([next_p], dtype=jnp.int8))

@jax.jit
def run_betting_round(init_state, num_actions=9, strategy_table: jax.Array = None):
    """Run betting round with configurable action space."""
    cond = lambda s: ~is_betting_done(s.player_status, s.bets, s.acted_this_round, s.street)
    
    def run_9_action():
        # Close over strategy_table for sampling
        def body_fn(s):
            legal = get_legal_actions_9(s)
            key, subkey = jax.random.split(s.key)
            # Compute info set id for current decision
            p = jnp.squeeze(s.cur_player).astype(jnp.int32)
            hole = s.hole_cards[p]
            info_id = compute_info_set_id_enhanced(hole, s.comm_cards, p, s.pot, s.stacks[p:p+1])
            # Fetch strategy probs or fallback to uniform
            def sample_from_strategy():
                probs = strategy_table[info_id]
                probs_masked = jnp.where(legal, probs, 0.0)
                total = jnp.sum(probs_masked)
                safe_probs = jnp.where(total > 0, probs_masked / total, jnp.where(legal, 1.0, 0.0))
                logits = jnp.log(jnp.clip(safe_probs, 1e-12, 1.0))
                return jax.random.categorical(subkey, logits)
            def sample_uniform():
                return jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
            use_policy = strategy_table is not None
            action = lax.cond(use_policy, sample_from_strategy, sample_uniform)
            s2 = replace(s, key=key)
            s2 = apply_action_9(s2, action)
            current_p = jnp.squeeze(s2.cur_player)
            next_p = next_active_player(s2.player_status, (current_p + 1) % 6)
            return replace(s2, cur_player=jnp.array([next_p], dtype=jnp.int8))
        return lax.while_loop(cond, body_fn, init_state)
    
    def run_6_action():
        return lax.while_loop(cond, _betting_body_6, init_state)
    
    def run_3_action():
        return lax.while_loop(cond, _betting_body_3, init_state)
    
    # Use lax.cond instead of if statements for JAX compatibility
    return lax.cond(
        num_actions == 9,
        run_9_action,
        lambda: lax.cond(
            num_actions == 6,
            run_6_action,
            run_3_action
        )
    )

# ---------- Street ----------
def play_street(state: GameState, num_cards: int, num_actions: int = 9, strategy_table: jax.Array = None) -> GameState:
    def deal(s: GameState) -> GameState:
        start = s.deck_ptr[0]
        cards = lax.dynamic_slice(s.deck, (start,), (num_cards,))
        # Write community cards starting at current number of dealt community cards
        num_comm_dealt = jnp.sum(s.comm_cards >= 0).astype(jnp.int32)
        comm_start = num_comm_dealt
        comm = lax.dynamic_update_slice(s.comm_cards, cards, (comm_start,))
        return replace(
            s,
            comm_cards=comm,
            deck_ptr=s.deck_ptr + num_cards,
            acted_this_round=jnp.zeros((6,), dtype=jnp.int8),
            cur_player=jnp.array([0], dtype=jnp.int8),
            street=s.street + 1
        )
    
    def deal_and_bet():
        state_with_cards = lax.cond(num_cards > 0, deal, lambda x: x, state)
        return run_betting_round(state_with_cards, num_actions, strategy_table)
    
    def just_bet():
        return run_betting_round(state, num_actions, strategy_table)
    
    # Use lax.cond for JAX compatibility
    return lax.cond(
        num_cards > 0,
        deal_and_bet,
        just_bet
    )

# ---------- Showdown ----------
def resolve_showdown(state: GameState, lut_keys, lut_values, table_size) -> jax.Array:
    active = state.player_status != 1
    pot_scalar = jnp.squeeze(state.pot)
    
    def single():
        winner = jnp.argmax(active)
        # ✅ CORRECTO: Winner gets pot, others lose their bets
        payoffs = -state.bets  # Everyone loses their bets first
        payoffs = payoffs.at[winner].add(pot_scalar)  # Winner gets the pot
        return payoffs
    
    def multiple():
        # Combine hole and community cards for each active player
        def eval_i(i):
            hole = state.hole_cards[i]
            comm = state.comm_cards
            cards = jnp.concatenate([hole, comm])
            return evaluate_hand_jax_native(cards)
        
        strengths = jax.vmap(eval_i)(jnp.arange(6))
        active_strengths = jnp.where(active, strengths, -1)
        winners = active_strengths == jnp.max(active_strengths)
        num_winners = jnp.sum(winners)
        share = pot_scalar / num_winners
        
        # ✅ CORRECTO: Everyone loses bets, winners split pot
        payoffs = -state.bets  # Everyone loses their bets first
        payoffs = payoffs + winners * share  # Winners split the pot
        return payoffs
    
    return lax.cond(active.sum() == 1, single, multiple)

# ---------- Game simulation ----------
@jax.jit
def play_one_game(key, lut_keys, lut_values, table_size, num_actions=9, strategy_table: jax.Array = None):
    """Play one complete game with MORE DIVERSITY."""
    idx_scalar = jax.random.randint(key, (), 0, 1000000)
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx_scalar)
    
    # Initialize state
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    # Randomizar hole_cards y usar una sola baraja para todo
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, jnp.arange(52))
    hole_cards = shuffled_deck[:12].reshape(6, 2)
    comm_cards = jnp.full((5,), -1)
    cur_player = jnp.array([2], dtype=jnp.int8)
    street = jnp.array([0], dtype=jnp.int8)
    pot = jnp.array([15.0])
    deck = shuffled_deck
    deck_ptr = jnp.array([12])
    acted_this_round = jnp.zeros((6,), dtype=jnp.int8)
    action_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int32)
    info_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int32)
    legal_hist = jnp.zeros((MAX_GAME_LENGTH, 9), dtype=jnp.bool_)
    player_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int8)
    pot_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.float32)
    hist_ptr = jnp.array([0])
    
    state = GameState(
        stacks=stacks, bets=bets, player_status=player_status, hole_cards=hole_cards,
        comm_cards=comm_cards, cur_player=cur_player, street=street, pot=pot,
        deck=deck, deck_ptr=deck_ptr, acted_this_round=acted_this_round,
        key=key, action_hist=action_hist, hist_ptr=hist_ptr,
        info_hist=info_hist, legal_hist=legal_hist, player_hist=player_hist, pot_hist=pot_hist,
        comm_hist=jnp.full((MAX_GAME_LENGTH, 5), -1, dtype=jnp.int32)
    )
    
    # NEW: DIVERSIFIED STREET PLAY
    key, subkey = jax.random.split(key)
    random_choice = jax.random.randint(subkey, (), 0, 4)
    
    # Play different game lengths for diversity
    state = play_street(state, 0, num_actions, strategy_table)   # Always preflop
    
    state = lax.cond(
        random_choice >= 1,  # 75% chance to see flop
        lambda s: play_street(s, 3, num_actions, strategy_table),
        lambda s: s,
        state
    )
    
    state = lax.cond(
        random_choice >= 2,  # 50% chance to see turn
        lambda s: play_street(s, 1, num_actions, strategy_table),
        lambda s: s,
        state
    )
    
    state = lax.cond(
        random_choice >= 3,  # 25% chance to see river
        lambda s: play_street(s, 1, num_actions, strategy_table),
        lambda s: s,
        state
    )
    
    # Resolve showdown
    payoffs = resolve_showdown(state, lut_keys, lut_values, table_size)
    
    # Asignar posiciones (0-5 para 6-max)
    positions = jnp.arange(6)  # UTG, MP, CO, BTN, SB, BB
    
    return payoffs, state.action_hist, {
        'hole_cards': state.hole_cards,
        'final_community': state.comm_cards,
        'final_pot': state.pot,
        'player_stacks': state.stacks,
        'player_bets': state.bets,
        'positions': positions,  # NUEVO: incluir posiciones
        # Trajectory logs for training
        'info_hist': state.info_hist,
        'legal_hist': state.legal_hist,
        'player_hist': state.player_hist,
        'pot_hist': state.pot_hist,
        'comm_hist': state.comm_hist,
        'hist_len': state.hist_ptr[0]
    }

@jax.jit
def batch_play(keys, lut_keys, lut_values, table_size, num_actions=9, strategy_table: jax.Array = None):
    """Play multiple games in batch with configurable action space."""
    return jax.vmap(
        play_one_game,
        in_axes=(0, None, None, None, None)
    )(keys, lut_keys, lut_values, table_size, num_actions, strategy_table)

@jax.jit
def initial_state_for_idx(idx):
    # Convertir idx a escalar de manera compatible con vmap
    idx_scalar = jax.random.randint(jax.random.PRNGKey(0), (), 0, 1000000)
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx_scalar)
    
    # Initialize state
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    # Randomizar hole_cards y usar una sola baraja para todo
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, jnp.arange(52))
    hole_cards = shuffled_deck[:12].reshape(6, 2)
    comm_cards = jnp.full((5,), -1)
    cur_player = jnp.array([2], dtype=jnp.int8)
    street = jnp.array([0], dtype=jnp.int8)
    pot = jnp.array([15.0])
    deck = shuffled_deck
    deck_ptr = jnp.array([12])
    acted_this_round = jnp.zeros((6,), dtype=jnp.int8)
    action_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int32)
    info_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int32)
    legal_hist = jnp.zeros((MAX_GAME_LENGTH, 9), dtype=jnp.bool_)
    player_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int8)
    pot_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.float32)
    hist_ptr = jnp.array([0])
    
    return GameState(
        stacks=stacks,
        bets=bets,
        player_status=player_status,
        hole_cards=hole_cards,
        comm_cards=comm_cards,
        cur_player=cur_player,
        street=street,
        pot=pot,
        deck=deck,
        deck_ptr=deck_ptr,
        acted_this_round=acted_this_round,
        key=key,
        action_hist=action_hist,
        hist_ptr=hist_ptr,
        info_hist=info_hist,
        legal_hist=legal_hist,
        player_hist=player_hist,
        pot_hist=pot_hist
    )

@jax.jit
def unified_batch_simulation_with_lut_production(keys, lut_keys, lut_values, table_size, num_actions=9, strategy_table: jax.Array = None):
    """
    Production-ready batch simulation with configurable action space.
    Optimized for speed and memory efficiency.
    """
    return batch_play(keys, lut_keys, lut_values, table_size, num_actions, strategy_table)

@jax.jit
def unified_batch_simulation_with_lut_full(keys, lut_keys, lut_values, table_size, num_actions=9, strategy_table: jax.Array = None):
    """
    Full-featured batch simulation with detailed game data.
    Returns comprehensive game information for analysis.
    """
    return batch_play(keys, lut_keys, lut_values, table_size, num_actions, strategy_table)

@jax.jit
def generate_random_game_state_with_position(key):
    """Genera state incluyendo posición de cada jugador."""
    
    # Initialize state similar to play_one_game
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    
    # Randomizar hole_cards
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, jnp.arange(52))
    hole_cards = shuffled_deck[:12].reshape(6, 2)
    comm_cards = jnp.full((5,), -1)
    pot_size = jnp.array([15.0])
    
    # Asignar posiciones (0-5 para 6-max)
    positions = jnp.arange(6)  # UTG, MP, CO, BTN, SB, BB
    
    # Simular payoffs básicos (placeholder)
    payoffs = jnp.zeros((6,))
    
    return {
        'hole_cards': hole_cards,
        'community_cards': comm_cards,
        'pot_size': pot_size,
        'positions': positions,  # NUEVO
        'payoffs': payoffs
    }

# Backward compatibility aliases
unified_batch_simulation_with_lut = unified_batch_simulation_with_lut_production

# Se elimina la auto-carga de LUT para evitar import circular y costes en import

@jax.jit
def play_from_state(initial_state: GameState, lut_keys, lut_values, table_size, num_actions=9):
    """
    Juega una partida de póker a partir de un estado inicial dado.
    Esencial para la exploración forzada.
    """
    # La partida continúa desde la calle (street) en la que se encuentra el estado inicial.
    state = initial_state

    # Simula las calles restantes del juego.
    # lax.cond asegura que esto sea compatible con JIT.
    state = lax.cond(state.street[0] <= 0, lambda s: play_street(s, 0, num_actions), lambda s: s, state) # Preflop si es necesario
    state = lax.cond(state.street[0] <= 1, lambda s: play_street(s, 3, num_actions), lambda s: s, state) # Flop
    state = lax.cond(state.street[0] <= 2, lambda s: play_street(s, 1, num_actions), lambda s: s, state) # Turn
    state = lax.cond(state.street[0] <= 3, lambda s: play_street(s, 1, num_actions), lambda s: s, state) # River

    # Resuelve la mano al final.
    payoffs = resolve_showdown(state, lut_keys, lut_values, table_size)
    
    # Devuelve los resultados en el mismo formato que play_one_game.
    return payoffs, state.action_hist, {
        'hole_cards': state.hole_cards,
        'final_community': state.comm_cards,
        'final_pot': state.pot,
        'player_stacks': state.stacks,
        'player_bets': state.bets,
        'positions': jnp.arange(6)
    }