"""
GPU Configuration for Modern CFR Poker AI
JAX 2025 XLA Optimization Flags for Maximum Performance
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# XLA Performance Flags (JAX 0.4.29 compatible)
XLA_FLAGS = (
    '--xla_gpu_enable_triton_gemm=true '
    '--xla_gpu_autotune_level=4 '
    '--xla_gpu_force_compilation_parallelism=0 '
)

def setup_gpu_optimization():
    """Setup XLA flags for maximum GPU performance"""
    os.environ['XLA_FLAGS'] = XLA_FLAGS
    
    # Additional JAX configuration
    os.environ['JAX_ENABLE_X64'] = 'False'  # Use 32-bit by default
    os.environ['JAX_PLATFORMS'] = 'gpu,cpu'  # Prefer GPU
    
    logger.info("GPU optimization flags configured")
    logger.info(f"XLA_FLAGS: {XLA_FLAGS}")

def get_device_info() -> Dict[str, Any]:
    """Get information about available devices"""
    devices = jax.devices()
    
    info = {
        'num_devices': len(devices),
        'devices': [str(device) for device in devices],
        'local_devices': len(jax.local_devices()),
        'device_count': jax.device_count(),
        'platform': jax.default_backend(),
    }
    
    return info

def configure_memory_allocation():
    """Configure JAX memory allocation for optimal performance"""
    # JAX pre-allocates 75% of GPU memory by default
    # This is usually optimal for our use case
    
    # For debugging memory issues, you can use:
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    logger.info("Memory allocation configured (default: 75% pre-allocation)")

def setup_mixed_precision():
    """Setup mixed precision training configuration"""
    # bfloat16 for computation, float32 for numerics
    precision_config = {
        'computation_dtype': jnp.bfloat16,
        'accumulation_dtype': jnp.float32,
        'gradient_dtype': jnp.float32,
        'parameter_dtype': jnp.float32,
    }
    
    logger.info("Mixed precision configured: bfloat16 computation, float32 numerics")
    return precision_config

def init_gpu_environment():
    """Initialize complete GPU environment for training"""
    setup_gpu_optimization()
    configure_memory_allocation()
    precision_config = setup_mixed_precision()
    device_info = get_device_info()
    
    logger.info("=== GPU Environment Initialized ===")
    logger.info(f"Platform: {device_info['platform']}")
    logger.info(f"Devices: {device_info['num_devices']}")
    logger.info(f"Local devices: {device_info['local_devices']}")
    
    return {
        'device_info': device_info,
        'precision_config': precision_config,
    }

# Auto-initialize when imported
if __name__ != '__main__':
    try:
        init_gpu_environment()
    except Exception as e:
        logger.warning(f"GPU environment initialization failed: {e}")
        logger.warning("Falling back to CPU-only mode") 