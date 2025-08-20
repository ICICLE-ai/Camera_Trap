"""
Utilities module for ICICLE-Benchmark V2
"""

# Import main utility classes and functions
from .logging import ICICLELogger, icicle_logger, setup_logging
from .metrics import MetricsCalculator
from .gpu import GPUManager, GPUCleanupContext
from .seed import set_seed
from .paths import (
    setup_experiment_directories, 
    get_checkpoint_directories,
    validate_camera_data,
    get_results_file_path,
    get_log_file_path
)
from .config import (
    load_config, 
    merge_configs, 
    update_config_with_args,
    validate_config,
    get_mode_type,
    get_config_summary
)
from .results import ResultsManager, aggregate_results

# Version info
__version__ = "2.0.0"

# Default exports
__all__ = [
    # Logging
    'ICICLELogger',
    'icicle_logger', 
    'setup_logging',
    
    # Metrics
    'MetricsCalculator',
    
    # GPU management
    'GPUManager',
    'GPUCleanupContext',
    
    # Random seed
    'set_seed',
    
    # Path management
    'setup_experiment_directories',
    'get_checkpoint_directories', 
    'validate_camera_data',
    'get_results_file_path',
    'get_log_file_path',
    
    # Configuration
    'load_config',
    'merge_configs',
    'update_config_with_args', 
    'validate_config',
    'get_mode_type',
    'get_config_summary',
    
    # Results management
    'ResultsManager',
    'aggregate_results'
]
