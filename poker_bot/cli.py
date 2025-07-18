# poker_bot/cli.py

"""
Clean Command Line Interface for Aequus JAX Poker AI
Organized, English-only, focused on core functionality
"""

import logging
import os
import click
import jax
from pathlib import Path

from .core.trainer import create_trainer, TrainerConfig
from .core.validation import detailed_validation, quick_validation
from .core.bucketing import validate_bucketing_system
from .bot import PokerBot

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Aequus JAX: High-Performance Poker AI with JAX acceleration"""
    # Display JAX information
    click.echo(f"🎯 Aequus JAX Poker AI v2.0.0")
    click.echo(f"JAX version: {jax.__version__}")
    click.echo(f"JAX devices: {len(jax.devices())} ({jax.devices()})")
    click.echo()

@cli.command()
@click.option('--config', '-c', 
              default='config/training_config.yaml',
              help='Path to training configuration file')
@click.option('--iterations', '-i', 
              default=None, type=int,
              help='Number of training iterations (overrides config)')
@click.option('--save-path', '-s',
              default='models/aequus_model',
              help='Base path for saving models')
@click.option('--validate', is_flag=True,
              help='Run validation before and after training')
@click.option('--resume',
              help='Resume training from checkpoint file or directory')
def train(config: str, iterations: int, save_path: str, validate: bool, resume: str):
    """Train the poker AI using CFR algorithm"""
    
    click.echo(f"🚀 Starting Aequus training session...")
    click.echo(f"Config: {config}")
    click.echo(f"Save path: {save_path}")
    if resume:
        click.echo(f"Resume from: {resume}")
    
    try:
        # Create trainer
        if resume:
            # Resume mode - create trainer from checkpoint
            if os.path.isfile(resume):
                # Direct checkpoint file
                trainer = create_trainer()
                trainer.load_model(resume)
                click.echo(f"✅ Resuming from checkpoint: {resume}")
            elif os.path.isdir(resume):
                # Directory - find latest checkpoint
                trainer = create_trainer()
                checkpoint = trainer.find_latest_checkpoint(resume)
                if checkpoint:
                    trainer.load_model(checkpoint)
                    click.echo(f"✅ Resuming from latest checkpoint: {checkpoint}")
                else:
                    raise click.ClickException(f"No checkpoints found in directory: {resume}")
            else:
                raise click.ClickException(f"Resume path not found: {resume}")
        else:
            # Fresh training mode
            if os.path.exists(config):
                trainer = create_trainer(config)
                click.echo(f"✅ Loaded configuration from {config}")
            else:
                trainer = create_trainer()
                click.echo(f"⚠️ Config not found, using defaults")
     
        # Override iterations if specified (only for fresh training)
        if iterations and not resume:
            click.echo(f"🔧 Overriding iterations: {iterations}")
        elif not iterations and not resume:
            iterations = 1000  # Default for fresh training
        elif resume:
            # For resume, iterations is required
            if not iterations:
                raise click.ClickException("--iterations is required when using --resume")
            click.echo(f"🔧 Additional iterations: {iterations}")
     
        # Pre-training validation
        if validate:
            click.echo(f"\n🔍 Running pre-training validation...")
            validation_passed = quick_validation(trainer.strategy)
            if not validation_passed:
                click.echo(f"❌ Pre-training validation failed!")
                detailed_validation(trainer.strategy)
                return
            click.echo(f"✅ Pre-training validation passed")
     
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
     
        # Run training
        if resume:
            click.echo(f"\n🔄 Resuming CFR training...")
            stats = trainer.resume_training(resume, iterations, save_path)
        else:
            click.echo(f"\n🎯 Starting fresh CFR training...")
            stats = trainer.train(iterations, save_path)
        
        # Post-training validation
        if validate:
            click.echo(f"\n🔍 Running post-training validation...")
            results = detailed_validation(trainer.strategy)
            if results.passed:
                click.echo(f"🎉 Training completed successfully!")
            else:
                click.echo(f"⚠️ Training completed with issues")
        
        # Display final statistics
        click.echo(f"\n📊 Training Statistics:")
        click.echo(f"   Iterations: {stats['iterations_completed']:,}")
        click.echo(f"   Time: {stats['total_time']:.1f}s")
        click.echo(f"   Speed: {stats['iterations_per_second']:.1f} iter/s")
        click.echo(f"   Final entropy: {stats['final_strategy_entropy']:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        logger.exception("Training error")
        raise click.ClickException(f"Training failed: {e}")

@cli.command()
@click.option('--model', '-m', required=True,
              help='Path to trained model file (.pkl)')
@click.option('--games', '-g', default=10,
              help='Number of test games to play')
def play(model: str, games: int):
    """Test the trained bot by playing games"""
    
    if not os.path.exists(model):
        raise click.ClickException(f"Model not found: {model}")
    
    click.echo(f"🤖 Loading bot from {model}...")
    
    try:
        bot = PokerBot(model_path=model)
        click.echo(f"✅ Bot loaded successfully!")
        
        # Simple test games
        click.echo(f"\n🎮 Playing {games} test games...")
        
        for i in range(games):
            # Create a simple test game state
            test_state = {
                'player_id': 0,
                'hole_cards': [i * 2, i * 2 + 1],  # Different cards each game
                'community_cards': [10, 11, 12, 13, 14],
                'pot_size': 50.0,
                'position': i % 6
            }
            
            action = bot.get_action(test_state)
            click.echo(f"   Game {i+1}: Action = {action}")
        
        click.echo(f"🎉 Test games completed!")
        
    except Exception as e:
        click.echo(f"❌ Bot testing failed: {e}")
        logger.exception("Bot testing error")
        raise click.ClickException(f"Bot testing failed: {e}")

@cli.command()
@click.option('--model', '-m',
              help='Path to trained model to validate (optional)')
@click.option('--detailed', is_flag=True,
              help='Run detailed validation with full reporting')
def validate(model: str, detailed: bool):
    """Validate system components and trained models"""
    
    click.echo(f"🔍 Running Aequus validation suite...")
    
    try:
        # Validate bucketing system
        click.echo(f"\n🧪 Validating bucketing system...")
        bucketing_ok = validate_bucketing_system()
        if bucketing_ok:
            click.echo(f"✅ Bucketing system validation passed")
        else:
            click.echo(f"❌ Bucketing system validation failed")
        
        # Validate trained model if provided
        if model:
            if not os.path.exists(model):
                raise click.ClickException(f"Model not found: {model}")
            
            click.echo(f"\n🤖 Validating trained model: {model}")
            bot = PokerBot(model_path=model)
            
            # Load strategy for validation
            import pickle
            with open(model, 'rb') as f:
                model_data = pickle.load(f)
            strategy = model_data['strategy']
            
            if detailed:
                results = detailed_validation(strategy)
                if results.passed:
                    click.echo(f"🎉 Model validation passed!")
                else:
                    click.echo(f"⚠️ Model validation failed with {len(results.errors)} errors")
            else:
                passed = quick_validation(strategy)
                if passed:
                    click.echo(f"✅ Quick model validation passed")
                else:
                    click.echo(f"❌ Quick model validation failed")
        
        # System health check
        click.echo(f"\n💻 System health check...")
        click.echo(f"   JAX devices: {len(jax.devices())}")
        click.echo(f"   JAX backend: {jax.default_backend()}")
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            click.echo(f"   Memory: {memory.percent:.1f}% used")
        except ImportError:
            click.echo(f"   Memory: psutil not available")
        
        click.echo(f"\n🎉 Validation suite completed!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        logger.exception("Validation error")
        raise click.ClickException(f"Validation failed: {e}")

@cli.command()
@click.option('--config', '-c',
              default='config/training_config.yaml',
              help='Configuration file to check')
def info(config: str):
    """Display system and configuration information"""
    
    click.echo(f"📋 Aequus JAX System Information")
    click.echo(f"=" * 40)
    
    # JAX information
    click.echo(f"\n🔧 JAX Configuration:")
    click.echo(f"   Version: {jax.__version__}")
    click.echo(f"   Backend: {jax.default_backend()}")
    click.echo(f"   Devices: {jax.devices()}")
    click.echo(f"   Device count: {jax.device_count()}")
    
    # Configuration information
    if os.path.exists(config):
        click.echo(f"\n⚙️ Training Configuration:")
        try:
            import yaml
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for key, value in config_data.items():
                click.echo(f"   {key}: {value}")
                
        except Exception as e:
            click.echo(f"   Error reading config: {e}")
    else:
        click.echo(f"\n⚠️ Configuration file not found: {config}")
        click.echo(f"   Will use default configuration")
    
    # System resources
    try:
        import psutil
        click.echo(f"\n💻 System Resources:")
        click.echo(f"   CPU cores: {psutil.cpu_count()}")
        
        memory = psutil.virtual_memory()
        click.echo(f"   Memory: {memory.total // (1024**3):.1f} GB total")
        click.echo(f"   Memory: {memory.percent:.1f}% used")
        
    except ImportError:
        click.echo(f"\n💻 System Resources: psutil not available")
    
    # Project structure
    click.echo(f"\n📁 Project Structure:")
    project_files = [
        'poker_bot/core/trainer.py',
        'poker_bot/core/bucketing.py', 
        'poker_bot/core/validation.py',
        'poker_bot/core/full_game_engine.py',
        'poker_bot/bot.py',
        'config/training_config.yaml'
    ]
    
    for file_path in project_files:
        if os.path.exists(file_path):
            click.echo(f"   ✅ {file_path}")
        else:
            click.echo(f"   ❌ {file_path}")

@cli.command()
@click.option('--base-config', 
              default='config/training_config.yaml',
              help='Base configuration file')
@click.option('--output', '-o',
              default='config/custom_config.yaml',
              help='Output path for new configuration')
@click.option('--batch-size', type=int, help='Batch size')
@click.option('--iterations', type=int, help='Number of iterations')
@click.option('--learning-rate', type=float, help='Learning rate')
def configure(base_config: str, output: str, batch_size: int, 
              iterations: int, learning_rate: float):
    """Create or modify training configuration"""
    
    click.echo(f"⚙️ Creating custom configuration...")
    
    try:
        # Load base configuration
        if os.path.exists(base_config):
            import yaml
            with open(base_config, 'r') as f:
                config = yaml.safe_load(f)
            click.echo(f"📂 Loaded base config from {base_config}")
        else:
            config = {}
            click.echo(f"⚠️ Base config not found, creating new config")
        
        # Apply overrides
        if batch_size:
            config['batch_size'] = batch_size
            click.echo(f"   Set batch_size: {batch_size}")
        
        if iterations:
            config['num_iterations'] = iterations
            click.echo(f"   Set num_iterations: {iterations}")
        
        if learning_rate:
            config['learning_rate'] = learning_rate
            click.echo(f"   Set learning_rate: {learning_rate}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        # Save new configuration
        import yaml
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        click.echo(f"💾 Configuration saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Configuration creation failed: {e}")
        raise click.ClickException(f"Configuration creation failed: {e}")

if __name__ == '__main__':
    cli()