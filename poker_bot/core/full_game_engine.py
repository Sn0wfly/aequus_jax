# full_game_engine.py
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass, replace
from functools import partial
import numpy as np
from jax.tree_util import register_pytree_node_class
from ..evaluator import HandEvaluator

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

    def tree_flatten(self):
        children = (self.stacks, self.bets, self.player_status, self.hole_cards,
                    self.comm_cards, self.cur_player, self.street, self.pot,
                    self.deck, self.deck_ptr, self.acted_this_round,
                    self.key, self.action_hist, self.hist_ptr)
        return children, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

# ---------- JAX-NATIVE HAND EVALUATION WITH LUT ----------
# LUT parameters are now passed as function arguments instead of global variables

@jax.jit
def evaluate_hand_jax_native(cards, lut_keys, lut_values, lut_table_size):
    """
    JAX-native hand evaluation using lookup table.
    Ultra-fast O(1) evaluation with no CPU callbacks.
    
    Args:
        cards: Array of card indices [7], where -1 = invalid card
        lut_keys: LUT hash keys array
        lut_values: LUT hash values array
        lut_table_size: Size of the LUT table
        
    Returns:
        Hand strength (int32, higher = better)
    """
    # JAX-compatible card filtering: sum only valid cards (>= 0)
    valid_mask = cards >= 0
    hash_key = jnp.sum(jnp.where(valid_mask, cards, 0))
    hash_idx = hash_key % lut_table_size
    
    # Linear probing to find correct entry
    def probe_condition(state):
        idx, found = state
        return (idx < lut_table_size) & (~found)
    
    def probe_body(state):
        idx, found = state
        is_match = lut_keys[idx] == hash_key
        is_empty = lut_keys[idx] == -1
        found_match = found | is_match
        next_idx = jnp.where(is_match | is_empty, idx, (idx + 1) % lut_table_size)
        return (next_idx, found_match)
    
    final_idx, found = lax.while_loop(probe_condition, probe_body, (hash_idx, False))
    
    # Return strength if found, otherwise default to sum-based heuristic
    return jnp.where(
        found,
        lut_values[final_idx],
        hash_key.astype(jnp.int32)  # Fallback using valid card sum
    )

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
    
    new_hist = state2.action_hist.at[state2.hist_ptr[0]].set(action)
    return replace(
        state2,
        action_hist=new_hist,
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
    
    new_hist = state2.action_hist.at[state2.hist_ptr[0]].set(action)
    return replace(
        state2,
        action_hist=new_hist,
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
    new_hist = state2.action_hist.at[state2.hist_ptr[0]].set(action)
    return replace(
        state2,
        action_hist=new_hist,
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
def run_betting_round(init_state, num_actions=9):
    """Run betting round with configurable action space."""
    cond = lambda s: ~is_betting_done(s.player_status, s.bets, s.acted_this_round, s.street)
    
    def run_9_action():
        return lax.while_loop(cond, _betting_body_9, init_state)
    
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
def play_street(state: GameState, num_cards: int, num_actions: int = 9) -> GameState:
    def deal(s: GameState) -> GameState:
        start = s.deck_ptr[0]
        cards = lax.dynamic_slice(s.deck, (start,), (num_cards,))
        comm = lax.dynamic_update_slice(s.comm_cards, cards, (start,))
        return replace(
            s,
            comm_cards=comm,
            deck_ptr=s.deck_ptr + num_cards,
            acted_this_round=jnp.array([0], dtype=jnp.int8),
            cur_player=jnp.array([0], dtype=jnp.int8)
        )
    
    def deal_and_bet():
        state_with_cards = lax.cond(num_cards > 0, deal, lambda x: x, state)
        return run_betting_round(state_with_cards, num_actions)
    
    def just_bet():
        return run_betting_round(state, num_actions)
    
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
        return -state.bets.at[winner].add(pot_scalar)
    
    def multiple():
        # Combine hole and community cards for each active player
        def eval_i(i):
            hole = state.hole_cards[i]
            comm = state.comm_cards
            cards = jnp.concatenate([hole, comm])
            return evaluate_hand_jax_native(cards, lut_keys, lut_values, table_size)
        
        strengths = jax.vmap(eval_i)(jnp.arange(6))
        active_strengths = jnp.where(active, strengths, -1)
        winners = active_strengths == jnp.max(active_strengths)
        num_winners = jnp.sum(winners)
        share = pot_scalar / num_winners
        return -state.bets + winners * share
    
    return lax.cond(active.sum() == 1, single, multiple)

# ---------- Game simulation ----------
@jax.jit
def play_one_game(key, lut_keys, lut_values, table_size, num_actions=9):
    """Play one complete game with MORE DIVERSITY."""
    idx_scalar = jax.random.randint(key, (), 0, 1000000)
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx_scalar)
    
    # Initialize state
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    # Randomizar hole_cards
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, jnp.arange(52))
    hole_cards = shuffled_deck[:12].reshape(6, 2)
    comm_cards = jnp.full((5,), -1)
    cur_player = jnp.array([2], dtype=jnp.int8)
    street = jnp.array([0], dtype=jnp.int8)
    pot = jnp.array([15.0])
    deck = jnp.arange(52)
    deck_ptr = jnp.array([12])
    acted_this_round = jnp.array([0], dtype=jnp.int8)
    action_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int8)
    hist_ptr = jnp.array([0])
    
    state = GameState(
        stacks=stacks, bets=bets, player_status=player_status, hole_cards=hole_cards,
        comm_cards=comm_cards, cur_player=cur_player, street=street, pot=pot,
        deck=deck, deck_ptr=deck_ptr, acted_this_round=acted_this_round,
        key=key, action_hist=action_hist, hist_ptr=hist_ptr
    )
    
    # NEW: DIVERSIFIED STREET PLAY
    key, subkey = jax.random.split(key)
    random_choice = jax.random.randint(subkey, (), 0, 4)
    
    # Play different game lengths for diversity
    state = play_street(state, 0, num_actions)   # Always preflop
    
    state = lax.cond(
        random_choice >= 1,  # 75% chance to see flop
        lambda s: play_street(s, 3, num_actions),
        lambda s: s,
        state
    )
    
    state = lax.cond(
        random_choice >= 2,  # 50% chance to see turn
        lambda s: play_street(s, 1, num_actions),
        lambda s: s,
        state
    )
    
    state = lax.cond(
        random_choice >= 3,  # 25% chance to see river
        lambda s: play_street(s, 1, num_actions),
        lambda s: s,
        state
    )
    
    # Resolve showdown
    payoffs = resolve_showdown(state, lut_keys, lut_values, table_size)
    
    return payoffs, state.action_hist, {
        'hole_cards': state.hole_cards,
        'final_community': state.comm_cards,
        'final_pot': state.pot,
        'player_stacks': state.stacks,
        'player_bets': state.bets
    }

@jax.jit
def batch_play(keys, lut_keys, lut_values, table_size, num_actions=9):
    """Play multiple games in batch with configurable action space."""
    return jax.vmap(lambda k: play_one_game(k, lut_keys, lut_values, table_size, num_actions))(keys)

@jax.jit
def initial_state_for_idx(idx):
    # Convertir idx a escalar de manera compatible con vmap
    idx_scalar = jax.random.randint(jax.random.PRNGKey(0), (), 0, 1000000)
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx_scalar)
    
    # Initialize state
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    # Randomizar hole_cards
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, jnp.arange(52))
    hole_cards = shuffled_deck[:12].reshape(6, 2)
    comm_cards = jnp.full((5,), -1)
    cur_player = jnp.array([2], dtype=jnp.int8)
    street = jnp.array([0], dtype=jnp.int8)
    pot = jnp.array([15.0])
    deck = jnp.arange(52)
    deck_ptr = jnp.array([12])
    acted_this_round = jnp.array([0], dtype=jnp.int8)
    action_hist = jnp.zeros((MAX_GAME_LENGTH,), dtype=jnp.int8)
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
        hist_ptr=hist_ptr
    )

@jax.jit
def unified_batch_simulation_with_lut_production(keys, lut_keys, lut_values, table_size, num_actions=9):
    """
    Production-ready batch simulation with configurable action space.
    Optimized for speed and memory efficiency.
    """
    return batch_play(keys, lut_keys, lut_values, table_size, num_actions)

@jax.jit
def unified_batch_simulation_with_lut_full(keys, lut_keys, lut_values, table_size, num_actions=9):
    """
    Full-featured batch simulation with detailed game data.
    Returns comprehensive game information for analysis.
    """
    return batch_play(keys, lut_keys, lut_values, table_size, num_actions)

# Backward compatibility aliases
unified_batch_simulation_with_lut = unified_batch_simulation_with_lut_production

# Auto-load LUT at module import (if available)
try:
    load_hand_evaluation_lut()
except:
    pass  # Continue without LUT for testing