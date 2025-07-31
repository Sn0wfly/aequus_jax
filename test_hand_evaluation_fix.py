import jax.numpy as jnp
from poker_bot.core.starting_hands import classify_starting_hand
from poker_bot.core.board_analysis import analyze_hand_vs_board

def test_preflop_values():
    """Test that preflop hand values are now correct."""
    print("Testing preflop hand values...")
    
    # Test pocket jacks
    jj = jnp.array([39, 38])  # JJ
    strength = classify_starting_hand(jj)
    print(f"JJ strength: {strength:.3f}")
    assert strength > 0.69, f"JJ should be >0.69, got {strength}"
    
    # Test pocket aces  
    aa = jnp.array([48, 49])  # AA (correct encoding)
    strength = classify_starting_hand(aa)
    print(f"AA strength: {strength:.3f}")
    assert strength > 0.9, f"AA should be >0.9, got {strength}"
    
    # Test AK
    ak = jnp.array([48, 44])  # AK (correct encoding)
    strength = classify_starting_hand(ak)
    print(f"AK strength: {strength:.3f}")
    assert strength > 0.75, f"AK should be >0.75, got {strength}"
    
    # Test TT
    tt = jnp.array([35, 34])  # TT
    strength = classify_starting_hand(tt)
    print(f"TT strength: {strength:.3f}")
    assert strength > 0.6, f"TT should be >0.6, got {strength}"
    
    print("✅ Preflop values fixed!")

def test_postflop_values():
    """Test that postflop hand values are now correct."""
    print("\nTesting postflop hand values...")
    
    # Test AK on AKQ (two pair)
    ak = jnp.array([48, 44])  # AK (correct encoding)
    akq_board = jnp.array([48, 44, 40, -1, -1])  # AKQ
    strength = analyze_hand_vs_board(ak, akq_board)
    print(f"AK on AKQ (two pair): {strength:.3f}")
    assert strength > 0.8, f"AK on AKQ should be >0.8, got {strength}"
    
    # Test AK on 765 (overcards)
    ak = jnp.array([48, 44])  # AK (correct encoding)
    low_board = jnp.array([23, 19, 15, -1, -1])  # 765
    strength = analyze_hand_vs_board(ak, low_board)
    print(f"AK on 765 (overcards): {strength:.3f}")
    assert strength < 0.4, f"AK on 765 should be <0.4, got {strength}"
    
    # Test AA on K72 (overpair)
    aa = jnp.array([48, 49])  # AA (correct encoding)
    k72_board = jnp.array([44, 19, 11, -1, -1])  # K72
    strength = analyze_hand_vs_board(aa, k72_board)
    print(f"AA on K72 (overpair): {strength:.3f}")
    assert strength > 0.8, f"AA on K72 should be >0.8, got {strength}"
    
    # Test JJ on AKJ (set)
    jj = jnp.array([39, 38])  # JJ
    akj_board = jnp.array([48, 44, 36, -1, -1])  # AKJ (has a J)
    strength = analyze_hand_vs_board(jj, akj_board)
    print(f"JJ on AKJ (set): {strength:.3f}")
    assert strength > 0.8, f"JJ on AKJ should be >0.8, got {strength}"
    
    print("✅ Postflop values fixed!")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")
    
    # Test trash hand
    trash = jnp.array([3, 1])  # 72o
    strength = classify_starting_hand(trash)
    print(f"72o strength: {strength:.3f}")
    assert strength < 0.3, f"72o should be <0.3, got {strength}"
    
    # Test suited connectors
    suited_connector = jnp.array([16, 12])  # 65s (correct encoding)
    strength = classify_starting_hand(suited_connector)
    print(f"65s strength: {strength:.3f}")
    assert strength > 0.3, f"65s should be >0.3, got {strength}"
    
    print("✅ Edge cases working!")

if __name__ == "__main__":
    print("🎯 Testing Hand Evaluation Fixes")
    print("=" * 40)
    
    test_preflop_values()
    test_postflop_values()
    test_edge_cases()
    
    print("\n🎉 All hand evaluation fixes working!")
    print("✅ JJ now correctly valued at ~0.75")
    print("✅ AA now correctly valued at ~0.95")
    print("✅ Two pair now correctly valued at ~0.80")
    print("✅ Overcards now correctly valued at ~0.25") 