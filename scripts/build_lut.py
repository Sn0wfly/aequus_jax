#!/usr/bin/env python3
"""
Lookup Table Builder for Poker Hand Evaluation
Creates a comprehensive LUT for JAX-native hand evaluation without pure_callback

This script precalculates hand strengths for all possible combinations:
- 5-card hands: C(52,5) = 2,598,960 combinations
- 6-card hands: C(52,6) = 20,358,520 combinations  
- 7-card hands: C(52,7) = 133,784,560 combinations

Total: ~157M evaluations → ~600MB LUT file
"""

import numpy as np
import pickle
import os
import time
from itertools import combinations
from pathlib import Path
import sys

# Add parent directory to path to import poker_bot
sys.path.append(str(Path(__file__).parent.parent))

from poker_bot.evaluator import HandEvaluator

def build_comprehensive_lut():
    """
    Build comprehensive lookup table for all hand combinations.
    Uses card encoding: 0-51 where card = rank*4 + suit
    """
    print("🎯 Building Comprehensive Poker Hand Evaluation LUT...")
    print("=" * 60)
    
    evaluator = HandEvaluator()
    
    # Dictionary to store all evaluations
    # Key: sorted tuple of cards, Value: hand strength (int32)
    lut = {}
    
    total_combinations = 0
    processed = 0
    start_time = time.time()
    
    # Process 5, 6, and 7 card hands
    for num_cards in [5, 6, 7]:
        print(f"\n📋 Processing {num_cards}-card hands...")
        
        # Generate all combinations of this size
        combinations_iter = combinations(range(52), num_cards)
        
        # Count total combinations for this size
        from math import comb
        total_for_size = comb(52, num_cards)
        total_combinations += total_for_size
        
        print(f"   Total combinations: {total_for_size:,}")
        
        batch_size = 10000
        batch = []
        batch_count = 0
        
        for cards in combinations_iter:
            batch.append(cards)
            
            if len(batch) >= batch_size:
                # Process batch
                for card_combo in batch:
                    try:
                        strength = evaluator.evaluate_single(list(card_combo))
                        # Convert to "higher is better" and store as int32
                        lut[card_combo] = np.int32(7462 - strength)
                    except Exception as e:
                        print(f"Warning: Failed to evaluate {card_combo}: {e}")
                        lut[card_combo] = np.int32(0)
                
                processed += len(batch)
                batch = []
                batch_count += 1
                
                # Progress update
                if batch_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_combinations - processed) / rate if rate > 0 else 0
                    
                    print(f"   Progress: {processed:,}/{total_combinations:,} "
                          f"({100*processed/total_combinations:.1f}%) "
                          f"| Rate: {rate:.0f} eval/s | ETA: {eta/60:.1f}m")
        
        # Process remaining batch
        if batch:
            for card_combo in batch:
                try:
                    strength = evaluator.evaluate_single(list(card_combo))
                    lut[card_combo] = np.int32(7462 - strength)
                except Exception as e:
                    print(f"Warning: Failed to evaluate {card_combo}: {e}")
                    lut[card_combo] = np.int32(0)
            processed += len(batch)
    
    print(f"\n✅ LUT Construction Complete!")
    print(f"   Total evaluations: {len(lut):,}")
    print(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    return lut

def create_jax_compatible_arrays(lut):
    """
    Convert dictionary LUT to JAX-compatible arrays for ultra-fast lookup.
    
    Returns:
        (card_indices, strengths) where card_indices[i] maps to strengths[i]
    """
    print("\n🔧 Converting to JAX-compatible format...")
    
    # Sort by key for consistent indexing
    sorted_items = sorted(lut.items())
    
    # Create arrays
    card_combinations = []
    strengths = []
    
    for cards, strength in sorted_items:
        # Convert card tuple to padded array (max 7 cards)
        card_array = np.array(list(cards) + [-1] * (7 - len(cards)), dtype=np.int8)
        card_combinations.append(card_array)
        strengths.append(strength)
    
    card_indices = np.array(card_combinations, dtype=np.int8)
    strength_values = np.array(strengths, dtype=np.int32)
    
    print(f"   JAX arrays shape: cards{card_indices.shape}, strengths{strength_values.shape}")
    
    return card_indices, strength_values

def create_hash_based_lut(lut):
    """
    Create hash-based lookup table for O(1) access.
    Uses simple hash function: sum(cards) % table_size
    """
    print("\n⚡ Creating hash-based LUT for O(1) lookup...")
    
    # Find good table size (prime number larger than entries)
    table_size = len(lut) * 2 + 1
    while not is_prime(table_size):
        table_size += 2
    
    print(f"   Hash table size: {table_size:,}")
    
    # Initialize hash table
    hash_keys = np.full(table_size, -1, dtype=np.int64)  # -1 = empty
    hash_values = np.zeros(table_size, dtype=np.int32)
    
    collisions = 0
    
    for cards, strength in lut.items():
        # Simple hash: sum of cards mod table_size
        hash_key = sum(cards)
        hash_idx = hash_key % table_size
        
        # Linear probing for collisions
        while hash_keys[hash_idx] != -1:
            if hash_keys[hash_idx] == hash_key:
                break  # Found existing entry
            hash_idx = (hash_idx + 1) % table_size
            collisions += 1
        
        hash_keys[hash_idx] = hash_key
        hash_values[hash_idx] = strength
    
    print(f"   Hash collisions: {collisions:,}")
    print(f"   Load factor: {len(lut)/table_size:.2%}")
    
    return hash_keys, hash_values, table_size

def is_prime(n):
    """Simple primality test"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def save_lut(lut, hash_keys, hash_values, table_size, output_dir="data"):
    """Save LUT in multiple formats for flexibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving LUT to {output_dir}/...")
    
    # 1. Full dictionary (for debugging)
    dict_path = f"{output_dir}/hand_evaluations_dict.pkl"
    with open(dict_path, 'wb') as f:
        pickle.dump(lut, f)
    dict_size = os.path.getsize(dict_path) / (1024*1024)
    print(f"   Dictionary: {dict_path} ({dict_size:.1f} MB)")
    
    # 2. Hash table (for production)
    hash_data = {
        'hash_keys': hash_keys,
        'hash_values': hash_values, 
        'table_size': table_size,
        'num_entries': len(lut)
    }
    hash_path = f"{output_dir}/hand_evaluations_hash.pkl"
    with open(hash_path, 'wb') as f:
        pickle.dump(hash_data, f)
    hash_size = os.path.getsize(hash_path) / (1024*1024)
    print(f"   Hash table: {hash_path} ({hash_size:.1f} MB)")
    
    # 3. Metadata
    metadata = {
        'total_entries': len(lut),
        'card_combinations': {
            '5_card': sum(1 for k in lut.keys() if len(k) == 5),
            '6_card': sum(1 for k in lut.keys() if len(k) == 6), 
            '7_card': sum(1 for k in lut.keys() if len(k) == 7),
        },
        'table_size': table_size,
        'format_version': '1.0'
    }
    metadata_path = f"{output_dir}/lut_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"   Metadata: {metadata_path}")
    print(f"\n✅ LUT saved successfully!")
    print(f"   Total disk usage: {(dict_size + hash_size):.1f} MB")
    
    return hash_path

def main():
    """Main LUT building process"""
    print("🚀 Aequus JAX Poker AI - LUT Builder")
    print("Building comprehensive hand evaluation lookup table...")
    print()
    
    try:
        # Build the lookup table
        lut = build_comprehensive_lut()
        
        # Create hash-based structure for fast lookup
        hash_keys, hash_values, table_size = create_hash_based_lut(lut)
        
        # Save everything
        lut_path = save_lut(lut, hash_keys, hash_values, table_size)
        
        print(f"\n🎯 SUCCESS! Lookup table ready for production use.")
        print(f"   Load in game engine with: pickle.load(open('{lut_path}', 'rb'))")
        print()
        print("Next step: Update full_game_engine.py to use this LUT!")
        
    except Exception as e:
        print(f"\n❌ Error building LUT: {e}")
        raise

if __name__ == "__main__":
    main() 