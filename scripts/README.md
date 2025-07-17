# Scripts Directory

This directory contains utility scripts for the Aequus JAX Poker AI project.

## Lookup Table (LUT) Builder

### Overview
The `build_lut.py` script creates a comprehensive lookup table for poker hand evaluation, enabling ultra-fast JAX-native evaluation without CPU-GPU synchronization bottlenecks.

### Usage

#### 1. Build the LUT (one-time setup):
```bash
python scripts/build_lut.py
```

**Expected output:**
- **Processing time**: ~30-60 minutes on modern CPU
- **Total evaluations**: ~157 million hand combinations
- **Output files**: 
  - `data/hand_evaluations_hash.pkl` (~300MB) - Production hash table
  - `data/hand_evaluations_dict.pkl` (~400MB) - Full dictionary for debugging
  - `data/lut_metadata.pkl` - Metadata and statistics

#### 2. The LUT is automatically loaded when importing the game engine:
```python
from poker_bot.core import load_hand_evaluation_lut

# Manual loading (optional)
success = load_hand_evaluation_lut("data/hand_evaluations_hash.pkl")
if success:
    print("✅ Real poker evaluation active")
else:
    print("⚠️ Using fallback heuristic evaluation")
```

#### 3. Training with real evaluation:
```bash
# After building LUT, training will automatically use real hand evaluation
python -m poker_bot.cli train --config config/training_config.yaml --iterations 1000 --save-path models/real_evaluation
```

### Performance Impact

| Mode | Evaluation | Speed | GPU Usage | Accuracy |
|------|------------|-------|-----------|-----------|
| **Before** | pure_callback | 5 iter/s | 5% | ✅ Real |
| **Test** | Sum heuristic | 271 iter/s | 85%+ | ❌ Fake |
| **Production** | JAX-native LUT | 250+ iter/s | 85%+ | ✅ Real |

### Architecture

```
Training Loop:
├── Game Simulation (GPU) ─── JAX-native ───┐
├── Hand Evaluation (GPU) ─── LUT O(1) ─────┼─── Pure GPU Pipeline
├── Regret Updates (GPU) ─── Vectorized ────┘
└── No CPU-GPU sync needed! 🚀
```

### Troubleshooting

**"LUT file not found"**: Run `python scripts/build_lut.py` first

**"Memory error during build"**: Reduce batch_size in build_lut.py (line 54)

**"Slow evaluation"**: Ensure LUT loaded successfully, check console for ✅ message

### Technical Details

- **Hash function**: `sum(cards) % table_size`
- **Collision resolution**: Linear probing
- **Memory usage**: ~600MB RAM for LUT storage
- **Lookup complexity**: O(1) average, O(k) worst case (k = max probe distance)
- **JAX compatibility**: 100% JIT-compilable, no Python callbacks 