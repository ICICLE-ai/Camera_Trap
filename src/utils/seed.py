"""
Random seed utilities
"""

import random
import numpy as np
import torch
import logging
import os

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # logger.info(f"Setting random seed to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    # Ensure deterministic algorithms are used where available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions
        pass

    # cuDNN and matmul settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable TF32 to avoid tiny numeric drift on Ampere+
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # Required by cuBLAS to be fully deterministic for some ops.
    # Must be set before first CUDA context is created.
    # If user didn't set it externally, set a safe default here.
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        # Small workspace usually sufficient and safer on constrained systems
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # Make Python hashing deterministic when iterating over sets/dicts
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    
    logger.debug("Random seed set for Python, NumPy, PyTorch, and CUDA")
