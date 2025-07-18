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
# Global LUT storage (loaded once at startup)
_LUT_HASH_KEYS = None
_LUT_HASH_VALUES = None
_LUT_TABLE_SIZE = None
_LUT_LOADED = False

def load_hand_evaluation_lut(lut_path="data/hand_evaluations_hash.pkl"):
    """
    Load hand evaluation LUT into global variables for JAX-native evaluation.
    Should be called once at startup.
    """
    global _LUT_HASH_KEYS, _LUT_HASH_VALUES, _LUT_TABLE_SIZE, _LUT_LOADED
    
    try:
        import pickle
        with open(lut_path, 'rb') as f:
            lut_data = pickle.load(f)
        
        _LUT_HASH_KEYS = jnp.array(lut_data['hash_keys'])
        _LUT_HASH_VALUES = jnp.array(lut_data['hash_values'])
        _LUT_TABLE_SIZE = lut_data['table_size']
        _LUT_LOADED = True
        
        print(f"✅ Hand evaluation LUT loaded: {len(_LUT_HASH_KEYS):,} entries")
        return True
        
    except FileNotFoundError:
        print(f"⚠️  LUT file not found: {lut_path}")
        print("   Run 'python scripts/build_lut.py' to create it")
        _LUT_LOADED = False
        return False
    except Exception as e:
        print(f"❌ Error loading LUT: {e}")
        _LUT_LOADED = False
        return False

@jax.jit  
def evaluate_hand_jax_native(cards):
    """
    JAX-native hand evaluation using lookup table.
    Ultra-fast O(1) evaluation with no CPU callbacks.
    
    Args:
        cards: Array of card indices [7], where -1 = invalid card
        
    Returns:
        Hand strength (int32, higher = better)
    """
    # JAX-compatible card filtering: sum only valid cards (>= 0)
    # Use jnp.where instead of boolean indexing
    valid_mask = cards >= 0
    hash_key = jnp.sum(jnp.where(valid_mask, cards, 0))
    hash_idx = hash_key % _LUT_TABLE_SIZE
    
    # Linear probing to find correct entry
    def probe_condition(state):
        idx, found = state
        return (idx < _LUT_TABLE_SIZE) & (~found)
    
    def probe_body(state):
        idx, found = state
        is_match = _LUT_HASH_KEYS[idx] == hash_key
        is_empty = _LUT_HASH_KEYS[idx] == -1
        found_match = found | is_match
        next_idx = jnp.where(is_match | is_empty, idx, (idx + 1) % _LUT_TABLE_SIZE)
        return (next_idx, found_match)
    
    final_idx, found = lax.while_loop(probe_condition, probe_body, (hash_idx, False))
    
    # Return strength if found, otherwise default to sum-based heuristic  
    return jnp.where(
        found,
        _LUT_HASH_VALUES[final_idx],
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
def get_legal_actions(state: GameState):
    mask = jnp.zeros(3, dtype=jnp.bool_)
    # Extraer índice del jugador de manera compatible con vmap
    p = jnp.squeeze(state.cur_player)  # Funciona tanto para escalares como arrays
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
def apply_action(state, action):
    p = jnp.squeeze(state.cur_player)  # Compatible con vmap
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

def step(state, action):
    return apply_action(state, action)

# ---------- Betting round ----------
def _betting_body(state):
    legal = get_legal_actions(state)
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
    state = replace(state, key=key)
    state = apply_action(state, action)
    current_p = jnp.squeeze(state.cur_player)  # Compatible con vmap
    next_p = next_active_player(state.player_status, (current_p + 1) % 6)
    return replace(state, cur_player=jnp.array([next_p], dtype=jnp.int8))

@jax.jit
def run_betting_round(init_state):
    cond = lambda s: ~is_betting_done(s.player_status, s.bets, s.acted_this_round, s.street)
    return lax.while_loop(cond, _betting_body, init_state)

# ---------- Street ----------
def play_street(state: GameState, num_cards: int) -> GameState:
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
    state = lax.cond(num_cards > 0, deal, lambda x: x, state)
    return run_betting_round(state)

# ---------- Showdown ----------
def resolve_showdown(state: GameState) -> jax.Array:
    active = state.player_status != 1
    pot_scalar = jnp.squeeze(state.pot)
    def single():
        winner = jnp.argmax(active)
        return -state.bets.at[winner].add(pot_scalar)
    def full():
        def eval_i(i):
            cards = jnp.concatenate([state.hole_cards[i], state.comm_cards])
            # JAX-NATIVE HAND EVALUATION: Use LUT if loaded, fallback to sum heuristic
            if _LUT_LOADED:
                return evaluate_hand_jax_native(cards)
            else:
                return jnp.sum(cards).astype(jnp.int32)  # Fallback for testing
        strengths = jnp.array([lax.cond(active[i], lambda: eval_i(i), lambda: 9999) for i in range(6)])
        best = jnp.min(strengths)
        winners = (strengths == best) & active
        share = pot_scalar / jnp.maximum(1, winners.sum())
        return -state.bets + winners * share
    can_show = (state.comm_cards != -1).sum() >= 5
    return lax.cond(active.sum() <= 1, single, lambda: lax.cond(can_show, full, single))

# ---------- Single game ----------
@jax.jit
def play_one_game(key):
    deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
    key, subkey = jax.random.split(key)
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].add(-5.0).at[1].add(-10.0)
    state = GameState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=deck[:12].reshape((6, 2)),
        comm_cards=jnp.full((5,), -1, dtype=jnp.int8),
        cur_player=jnp.array([2], dtype=jnp.int8),
        street=jnp.array([0], dtype=jnp.int8),
        pot=jnp.array([15.0]),
        deck=deck,
        deck_ptr=jnp.array([12], dtype=jnp.int8),
        acted_this_round=jnp.array([0], dtype=jnp.int8),
        key=subkey,
        action_hist=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        hist_ptr=jnp.array([0], dtype=jnp.int32)
    )
    state = play_street(state, 3)  # flop
    state = play_street(state, 1)  # turn
    state = play_street(state, 1)  # river
    payoffs = resolve_showdown(state)
    return payoffs, state.action_hist

# ---------- Batch API ----------
@jax.jit
def batch_play(keys):
    return jax.vmap(play_one_game)(keys)

@jax.jit
def initial_state_for_idx(idx):
    # Convertir idx a escalar de manera compatible con vmap
    idx_scalar = jnp.squeeze(idx) if hasattr(idx, 'squeeze') else idx
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx_scalar)
    # Devuelve solo el estado inicial, no payoffs ni historia
    deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
    key, subkey = jax.random.split(key)
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].add(-5.0).at[1].add(-10.0)
    state = GameState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=deck[:12].reshape((6, 2)),
        comm_cards=jnp.full((5,), -1, dtype=jnp.int8),
        cur_player=jnp.array([2], dtype=jnp.int8),
        street=jnp.array([0], dtype=jnp.int8),
        pot=jnp.array([15.0]),
        deck=deck,
        deck_ptr=jnp.array([12], dtype=jnp.int8),
        acted_this_round=jnp.array([0], dtype=jnp.int8),
        key=subkey,
        action_hist=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        hist_ptr=jnp.array([0], dtype=jnp.int32)
    )
    return state

# Agregar al final de full_game_engine.py

@jax.jit
def unified_batch_simulation(keys):
    """
    Extended batch simulation that returns game results for CFR training.
    
    Args:
        keys: Random keys for each game
        
    Returns:
        (payoffs, histories, game_results) tuple
    """
    # Run the basic batch simulation
    payoffs, histories = batch_play(keys)
    
    # Generate mock game results for training
    batch_size = keys.shape[0]  # JAX-compatible shape access
    
    # Fill hole cards from actual game simulation
    def extract_game_data(key):
        deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
        hole_cards = deck[:12].reshape((6, 2))
        community_cards = deck[12:17]
        return hole_cards, community_cards
    
    hole_cards_batch, community_batch = jax.vmap(extract_game_data)(keys)
    
    # Create all game results arrays directly - avoid dict modifications in JIT
    final_pot = jnp.abs(jnp.sum(payoffs, axis=1))
    player_stacks = jnp.ones((batch_size, 6)) * 100.0
    player_bets = jnp.abs(payoffs)
    
    # Create game results as final dictionary 
    game_results = {
        'hole_cards': hole_cards_batch,
        'final_community': community_batch, 
        'final_pot': final_pot,
        'player_stacks': player_stacks,
        'player_bets': player_bets
    }
    
    return payoffs, histories, game_results

# Auto-load LUT at module import (if available)
try:
    load_hand_evaluation_lut()
except:
    pass  # Continue without LUT for testing