# Aequus JAX Poker AI

A high-performance poker AI built with JAX acceleration, implementing Counterfactual Regret Minimization (CFR) for optimal poker strategy learning.

## 🚀 Features

- **JAX Acceleration**: Leverages JAX for GPU-accelerated training
- **CFR Algorithm**: Implements Counterfactual Regret Minimization for poker strategy optimization
- **Resume Training**: Robust checkpointing system for resuming interrupted training sessions
- **Flexible Configuration**: YAML-based configuration for easy customization
- **CLI Interface**: Clean command-line interface for training and evaluation

## 📋 Requirements

- Python 3.8+
- JAX (with GPU support recommended)
- PyYAML
- NumPy
- Click

## 🛠️ Installation

```bash
# Install dependencies
pip install -r config/requirements.txt

# Or install directly
pip install jax jaxlib pyyaml click numpy
```

## 🎯 Quick Start

### Fresh Training
```bash
# Basic training
python -m poker_bot.cli train --iterations 1000 --save-path models/my_model

# With custom config
python -m poker_bot.cli train --config config/training_config.yaml --iterations 5000
```

### Resume Training
```bash
# Resume from specific checkpoint
python -m poker_bot.cli train --resume checkpoints/model_iter_1000.pkl --iterations 500

# Resume from directory (finds latest checkpoint)
python -m poker_bot.cli train --resume checkpoints/ --iterations 1000
```

### Validation
```bash
# Validate trained model
python -m poker_bot.cli validate --model models/my_model_final.pkl --detailed

# Quick validation
python -m poker_bot.cli validate --model models/my_model_final.pkl
```

## ⚙️ Configuration

Training parameters can be customized via YAML configuration:

```yaml
# config/training_config.yaml
batch_size: 128
num_iterations: 1000
learning_rate: 0.01
save_interval: 100
log_interval: 50
```

## 🔧 CLI Commands

### `train`
Train the poker AI using CFR algorithm.

**Options:**
- `--config`: Path to training configuration file
- `--iterations`: Number of training iterations
- `--save-path`: Base path for saving models
- `--validate`: Run validation before and after training
- `--resume`: Resume training from checkpoint file or directory

### `validate`
Validate system components and trained models.

**Options:**
- `--model`: Path to trained model file
- `--detailed`: Run detailed validation with full reporting

### `play`
Test the trained bot by playing games.

**Options:**
- `--model`: Path to trained model file
- `--games`: Number of test games to play

## 📊 Training Process

1. **Initialization**: Trainer loads configuration and initializes regret/strategy tables
2. **CFR Training**: Iterative counterfactual regret minimization
3. **Checkpointing**: Periodic saves every `save_interval` iterations
4. **Resume Capability**: Can resume from any saved checkpoint

## 🔍 Architecture

- **Trainer**: Core CFR training logic in `poker_bot/core/trainer.py`
- **CLI**: Command-line interface in `poker_bot/cli.py`
- **Configuration**: YAML-based settings in `config/`
- **Models**: Saved models in `models/` directory

## 📝 License

MIT License - see LICENSE file for details