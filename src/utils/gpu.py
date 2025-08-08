"""
GPU management utilities
"""

import torch
import gc
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU memory and cleanup operations."""
    
    def __init__(self, enable_cleanup: bool = True):
        self.enable_cleanup = enable_cleanup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.enable_cleanup and torch.cuda.is_available():
            logger.info("GPU cleanup enabled")
            self.log_gpu_memory("Initial GPU state")
        elif not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU cleanup disabled")
            self.enable_cleanup = False
    
    def cleanup(self):
        """Clean GPU memory."""
        if not self.enable_cleanup or not torch.cuda.is_available():
            return
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Clear cache again
            torch.cuda.empty_cache()
            
            self.log_gpu_memory("After cleanup")
            
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
    
    def log_gpu_memory(self, prefix: str = ""):
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        try:
            allocated = torch.cuda.memory_allocated() / 1e9  # GB
            reserved = torch.cuda.memory_reserved() / 1e9   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
            
            logger.info(
                f"{prefix} GPU Memory - "
                f"Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, "
                f"Max Allocated: {max_allocated:.2f}GB"
            )
        except Exception as e:
            logger.warning(f"Failed to log GPU memory: {e}")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def context_cleanup(self):
        """Context manager for automatic cleanup."""
        return GPUCleanupContext(self)


class GPUCleanupContext:
    """Context manager for automatic GPU cleanup."""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
    
    def __enter__(self):
        return self.gpu_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gpu_manager.cleanup()
