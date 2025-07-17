"""
Memory Management for Modern CFR Poker AI
Gradient Checkpointing and JAX Memory Optimization
"""

import jax
import jax.numpy as jnp
from jax import checkpoint, remat
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from functools import partial, wraps
import gc
import psutil
import os

logger = logging.getLogger(__name__)

# Memory management configuration
class MemoryConfig:
    # Gradient checkpointing policy
    CHECKPOINT_POLICY = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    
    # Memory monitoring thresholds
    MEMORY_WARNING_THRESHOLD = 0.85  # 85% of available memory
    MEMORY_CRITICAL_THRESHOLD = 0.95  # 95% of available memory
    
    # Batch size adjustment for memory constraints
    MIN_BATCH_SIZE = 512
    MAX_BATCH_SIZE = 16384
    BATCH_SIZE_REDUCTION_FACTOR = 0.7

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        'process_memory_mb': memory_info.rss / 1024 / 1024,
        'system_memory_total_gb': system_memory.total / 1024 / 1024 / 1024,
        'system_memory_used_gb': system_memory.used / 1024 / 1024 / 1024,
        'system_memory_percent': system_memory.percent,
        'available_memory_gb': system_memory.available / 1024 / 1024 / 1024,
    }

def log_memory_usage(prefix: str = ""):
    """Log current memory usage"""
    stats = get_memory_usage()
    logger.info(f"{prefix}Memory Usage: Process={stats['process_memory_mb']:.1f}MB, "
                f"System={stats['system_memory_percent']:.1f}%, "
                f"Available={stats['available_memory_gb']:.1f}GB")

def check_memory_pressure() -> bool:
    """Check if system is under memory pressure"""
    stats = get_memory_usage()
    return stats['system_memory_percent'] > MemoryConfig.MEMORY_WARNING_THRESHOLD * 100

def emergency_memory_cleanup():
    """Emergency memory cleanup when under pressure"""
    logger.warning("Memory pressure detected, performing emergency cleanup")
    
    # Force garbage collection
    gc.collect()
    
    # Clear JAX compilation cache if available
    try:
        jax.clear_backends()
        logger.info("JAX backends cleared")
    except Exception as e:
        logger.warning(f"Failed to clear JAX backends: {e}")
    
    log_memory_usage("After cleanup: ")

def adaptive_batch_size(current_batch_size: int, memory_pressure: bool) -> int:
    """Adapt batch size based on memory pressure"""
    if memory_pressure:
        new_size = int(current_batch_size * MemoryConfig.BATCH_SIZE_REDUCTION_FACTOR)
        return max(new_size, MemoryConfig.MIN_BATCH_SIZE)
    else:
        # Gradually increase batch size if memory allows
        new_size = int(current_batch_size * 1.1)
        return min(new_size, MemoryConfig.MAX_BATCH_SIZE)

# Gradient checkpointing decorators
def checkpoint_wrapper(policy=None):
    """Decorator for gradient checkpointing"""
    if policy is None:
        policy = MemoryConfig.CHECKPOINT_POLICY
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            checkpointed_func = checkpoint(func, policy=policy)
            return checkpointed_func(*args, **kwargs)
        return wrapper
    return decorator

def remat_wrapper(policy=None):
    """Decorator for rematerialization"""
    if policy is None:
        policy = MemoryConfig.CHECKPOINT_POLICY
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            remat_func = remat(func, policy=policy)
            return remat_func(*args, **kwargs)
        return wrapper
    return decorator

# Memory-efficient training functions
@checkpoint_wrapper()
@jax.jit
def memory_efficient_cfr_step(params: Dict[str, jnp.ndarray],
                             state: Dict[str, Any],
                             batch: Dict[str, jnp.ndarray]) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Memory-efficient CFR training step with checkpointing"""
    # This would contain the actual CFR computation
    # For now, returning placeholders
    return params, state

@remat_wrapper()
@jax.jit
def memory_efficient_policy_evaluation(policy: Dict[str, jnp.ndarray],
                                     game_states: jnp.ndarray) -> jnp.ndarray:
    """Memory-efficient policy evaluation with rematerialization"""
    # This would contain the actual policy evaluation
    # For now, returning placeholder
    return jnp.zeros(len(game_states))

# Memory-aware data structures
class MemoryAwareCache:
    """Cache that automatically manages memory usage"""
    
    def __init__(self, max_size: int = 10000, memory_limit_mb: float = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with memory management"""
        # Check memory pressure before adding
        if check_memory_pressure():
            self._evict_items(0.5)  # Evict 50% of cache
        
        # Add or update item
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
    
    def _evict_items(self, fraction: float):
        """Evict a fraction of items"""
        num_to_evict = int(len(self.cache) * fraction)
        for _ in range(num_to_evict):
            if self.access_order:
                self._evict_lru()
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

# Memory-efficient tensor operations
@jax.jit
def memory_efficient_matmul(a: jnp.ndarray, b: jnp.ndarray, 
                           chunk_size: int = 1024) -> jnp.ndarray:
    """Memory-efficient matrix multiplication using chunking"""
    if a.shape[0] <= chunk_size:
        return jnp.dot(a, b)
    
    # Chunk the computation
    chunks = []
    for i in range(0, a.shape[0], chunk_size):
        chunk = jnp.dot(a[i:i+chunk_size], b)
        chunks.append(chunk)
    
    return jnp.concatenate(chunks, axis=0)

@jax.jit
def memory_efficient_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    """Memory-efficient softmax with numerical stability"""
    # Subtract max for numerical stability
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    stabilized_logits = logits - max_logits
    
    # Compute softmax
    exp_logits = jnp.exp(stabilized_logits)
    sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
    
    return exp_logits / sum_exp

@jax.jit
def memory_efficient_cross_entropy(logits: jnp.ndarray, 
                                  labels: jnp.ndarray) -> jnp.ndarray:
    """Memory-efficient cross-entropy loss"""
    # Use log_softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_probs, axis=-1)

# Memory monitoring context manager
class MemoryMonitor:
    """Context manager for monitoring memory usage"""
    
    def __init__(self, name: str, log_interval: int = 100):  # Changed from 1000 to 100
        self.name = name
        self.log_interval = log_interval
        self.step_count = 0
        self.start_memory = None
        self.total_enters = 0  # Track total context manager enters
    
    def __enter__(self):
        self.total_enters += 1
        # Only log every log_interval enters to prevent spam
        if self.total_enters % self.log_interval == 0:
            self.start_memory = get_memory_usage()
            log_memory_usage(f"[{self.name}] Start (step {self.total_enters}): ")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Only log exits that correspond to logged enters
        if self.total_enters % self.log_interval == 0 and self.start_memory is not None:
            end_memory = get_memory_usage()
            memory_diff = end_memory['process_memory_mb'] - self.start_memory['process_memory_mb']
            logger.info(f"[{self.name}] Memory change (step {self.total_enters}): {memory_diff:+.1f}MB")
        
        if exc_type is not None:
            logger.error(f"[{self.name}] Exception occurred: {exc_val}")
        
        self.start_memory = None
    
    def step(self):
        """Log memory usage at intervals"""
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            log_memory_usage(f"[{self.name}] Step {self.step_count}: ")
            
            # Check for memory pressure
            if check_memory_pressure():
                emergency_memory_cleanup()

# Optimized data loading with memory management
class MemoryEfficientDataLoader:
    """Data loader with memory-aware batching"""
    
    def __init__(self, data: np.ndarray, batch_size: int = 2048, 
                 adaptive_batching: bool = True):
        self.data = data
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size
        self.adaptive_batching = adaptive_batching
        self.index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self) -> jnp.ndarray:
        if self.index >= len(self.data):
            self.index = 0
            raise StopIteration
        
        # Adapt batch size based on memory pressure
        if self.adaptive_batching:
            memory_pressure = check_memory_pressure()
            self.current_batch_size = adaptive_batch_size(
                self.current_batch_size, memory_pressure
            )
        
        # Get next batch
        end_idx = min(self.index + self.current_batch_size, len(self.data))
        batch = self.data[self.index:end_idx]
        self.index = end_idx
        
        return jnp.array(batch)
    
    def reset(self):
        """Reset data loader"""
        self.index = 0
        self.current_batch_size = self.initial_batch_size

# Memory optimization utilities
def optimize_memory_usage():
    """Apply memory optimizations"""
    # Set JAX memory settings
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    
    # Enable memory debugging if needed
    if logger.getEffectiveLevel() == logging.DEBUG:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    
    logger.info("Memory optimizations applied")

def create_memory_efficient_trainer(config: Dict[str, Any]) -> Callable:
    """Create memory-efficient trainer with all optimizations"""
    
    @checkpoint_wrapper()
    def trainer(params, state, batch):
        # Training logic would go here
        return params, state
    
    return trainer

class AdaptiveBatchManager:
    """Adaptive batch size manager for memory-constrained training"""
    
    def __init__(self, base_batch_size: int = 2048, 
                 memory_threshold: float = 0.8,
                 min_batch_size: int = 512,
                 max_batch_size: int = 16384):
        self.base_batch_size = base_batch_size
        self.memory_threshold = memory_threshold
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = base_batch_size
        self.memory_history = []
        self.adjustment_factor = 0.8
        
        logger.info(f"AdaptiveBatchManager initialized with base_batch_size={base_batch_size}")
    
    def get_batch_size(self) -> int:
        """Get current adaptive batch size"""
        return self.current_batch_size
    
    def update_batch_size(self, memory_usage: Optional[Dict[str, float]] = None) -> int:
        """Update batch size based on memory usage"""
        if memory_usage is None:
            memory_usage = get_memory_usage()
        
        memory_percent = memory_usage['system_memory_percent'] / 100.0
        self.memory_history.append(memory_percent)
        
        # Keep only recent memory history
        if len(self.memory_history) > 10:
            self.memory_history.pop(0)
        
        # Calculate average memory usage
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        # Adjust batch size based on memory pressure
        if avg_memory > self.memory_threshold:
            # Reduce batch size
            new_batch_size = int(self.current_batch_size * self.adjustment_factor)
            self.current_batch_size = max(new_batch_size, self.min_batch_size)
            logger.info(f"Memory pressure detected ({avg_memory:.2f}), reducing batch size to {self.current_batch_size}")
        elif avg_memory < self.memory_threshold * 0.6:
            # Increase batch size if memory usage is low
            new_batch_size = int(self.current_batch_size / self.adjustment_factor)
            self.current_batch_size = min(new_batch_size, self.max_batch_size)
            logger.info(f"Low memory usage ({avg_memory:.2f}), increasing batch size to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def reset_batch_size(self):
        """Reset batch size to base value"""
        self.current_batch_size = self.base_batch_size
        self.memory_history = []
        logger.info(f"Batch size reset to {self.base_batch_size}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive batch manager statistics"""
        return {
            'current_batch_size': self.current_batch_size,
            'base_batch_size': self.base_batch_size,
            'memory_threshold': self.memory_threshold,
            'avg_memory_usage': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'memory_history_length': len(self.memory_history)
        }

# Initialize memory management
def init_memory_management():
    """Initialize memory management system"""
    optimize_memory_usage()
    log_memory_usage("Initial: ")
    
    # Create global memory monitor
    global_monitor = MemoryMonitor("Global", log_interval=5000)
    
    logger.info("Memory management initialized")
    return global_monitor

# Auto-initialize when imported
if __name__ != '__main__':
    init_memory_management() 