import jax
import jax.numpy as jnp
from poker_bot.core.bucketing import compute_info_set_id
import numpy as np

def test_stack_diversity_verification():
    """Test that real stack sizes are being used instead of hardcoded 1000.0."""
    
    print("🔍 Testing Stack Size Diversity Verification")
    print("=" * 60)
    
    # Test different stack sizes
    test_cases = [
        (300.0, "Short stack"),
        (800.0, "Medium stack"), 
        (1500.0, "Deep stack"),
        (200.0, "Very short stack")
    ]
    
    print("📊 Testing info set diversity with different stack sizes:")
    print()
    
    # Same hand, same position, same pot, but different stack sizes
    hole_cards = jnp.array([48, 49])  # AA
    community_cards = jnp.array([40, 41, 42, -1, -1])  # Flop
    player_idx = 0  # UTG
    pot_size = jnp.array([100.0])
    
    info_sets = []
    
    for stack_size, description in test_cases:
        info_set = compute_info_set_id(
            hole_cards, community_cards, player_idx, pot_size,
            stack_size=jnp.array([stack_size]), max_info_sets=10000
        )
        info_sets.append(int(info_set))
        print(f"   {description} ({stack_size} chips): Info Set {info_set}")
    
    print()
    
    # Check if all info sets are different
    unique_info_sets = len(set(info_sets))
    total_cases = len(test_cases)
    
    print(f"📈 Results:")
    print(f"   Total test cases: {total_cases}")
    print(f"   Unique info sets: {unique_info_sets}")
    print(f"   Diversity ratio: {unique_info_sets}/{total_cases} = {unique_info_sets/total_cases:.1%}")
    
    if unique_info_sets == total_cases:
        print("✅ SUCCESS: All different stack sizes produce different info sets!")
        print("✅ Stack diversity fix is working correctly!")
    else:
        print("❌ FAILURE: Some different stack sizes produce same info sets")
        print("❌ Stack diversity fix may not be working")
    
    print()
    print("🎯 Expected behavior:")
    print("   Before fix: All stack sizes → Same info set (hardcoded 1000.0)")
    print("   After fix: Different stack sizes → Different info sets")
    
    return unique_info_sets == total_cases

if __name__ == "__main__":
    test_stack_diversity_verification() 