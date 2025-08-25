#!/usr/bin/env python3
"""
Camera Trap Framework V2 - Main Entry Point

A clean, organized, and scalable camera trap evaluation framework.
Supports both config-based and argument-based execution with enhanced logging.

Usage:
    python main.py --camera APN_K024 --config configs/training/oracle.yaml
    python main.py --camera APN_K024 --config configs/training/accumulative.yaml
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Create logger first
logger = logging.getLogger(__name__)

# Import utilities with error handling
try:
    from src.utils import (
        icicle_logger, set_seed, GPUManager, MetricsCalculator,
        setup_experiment_directories, get_checkpoint_directories, validate_camera_data,
        load_config, update_config_with_args, validate_config, get_mode_type, get_config_summary,
        ResultsManager, setup_logging
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import all utils: {e}")
    UTILS_AVAILABLE = False

# Import core components with error handling
try:
    from src.config import ConfigManager
    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import ConfigManager: {e}")
    CONFIG_AVAILABLE = False

try:
    from src.models.factory import create_model
    MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import create_model: {e}")
    MODEL_AVAILABLE = False

try:
    from src.training import train_model_oracle, train_model_accumulative
    from src.main_utils import setup_model_and_data, evaluate_model_checkpoint_based
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import training modules: {e}")
    TRAINING_AVAILABLE = False

# Rest of the imports for data loading and evaluation
try:
    from src.data.loader import get_dataloaders
    from src.utils.paths import load_checkpoint_data
    DATA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import data modules: {e}")
    DATA_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Camera Trap Framework V2 - A comprehensive evaluation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --camera APN_K024 --config configs/training/oracle.yaml --epochs 5
  python main.py --camera APN_K024 --config configs/training/accumulative.yaml --epochs 2
        """
    )
    
    # Required arguments
    parser.add_argument('--camera', type=str, required=True,
                        help='Camera name (e.g., APN_K024)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    
    # Optional training arguments
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--train_val', action='store_true',
                        help='Enable validation during training')
    parser.add_argument('--train_test', action='store_true',
                        help='Enable testing during training')
    
    # Technical arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training/evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_and_validate_config(args):
    """Load and validate configuration."""
    if not CONFIG_AVAILABLE:
        raise RuntimeError("ConfigManager not available. Please check imports.")
    
    # Load config
    config = ConfigManager(args.config)
    config_dict = config.get_config()
    
    # Update with command line arguments
    config_dict = update_config_with_args(config_dict, args)
    
    # Validate configuration
    is_valid, issues = validate_config(config_dict, args)
    if not is_valid:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        raise ValueError("Invalid configuration")
    
    return config, config_dict


def setup_framework(args, config_dict):
    """Setup framework environment and logging."""
    # Set random seed
    set_seed(args.seed)
    
    # Setup GPU if available
    gpu_manager = GPUManager()
    gpu_info = gpu_manager.get_gpu_info()
    
    # Setup experiment directories
    log_dir = setup_experiment_directories(args, config_dict)
    
    # Setup logging
    setup_logging(log_dir)
    
    # Validate camera data
    validate_camera_data(args.camera)
    
    return log_dir


def run_training_mode(config, args):
    """Run training mode based on configuration."""
    if not TRAINING_AVAILABLE:
        raise RuntimeError("Training modules not available. Please check imports.")
    
    mode_type = get_mode_type(config)
    
    if mode_type == 'oracle':
        logger.info("Starting Oracle training...")
        trained_model = train_model_oracle(config, args)
    elif mode_type == 'accumulative':
        logger.info("Starting Accumulative training...")
        trained_model = train_model_accumulative(config, args)
    else:
        raise ValueError(f"Unknown training mode: {mode_type}")
    
    return trained_model


def run_evaluation_mode(config, args, trained_model=None):
    """Run evaluation mode."""
    logger.info("Starting evaluation phase...")
    
    # Run checkpoint-based evaluation
    avg_accuracy, results = evaluate_model_checkpoint_based(config, args, trained_model)
    
    return avg_accuracy, results


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load and validate configuration
        config, config_dict = load_and_validate_config(args)
        
        # Setup framework
        log_dir = setup_framework(args, config_dict)
        
        # Log setup information
        icicle_logger.log_setup_details(args, config_dict, log_dir)
        
        # Log dataset information
        icicle_logger.log_dataset_section(config_dict)
        
        # Determine mode and run appropriate workflow
        mode = config_dict.get('mode', 'train')
        
        if mode == 'train':
            # Training mode
            trained_model = run_training_mode(config_dict, args)
            
            # Run evaluation if model was trained successfully
            if trained_model is not None:
                avg_accuracy, results = run_evaluation_mode(config_dict, args, trained_model)
                
                # Log final results
                logger.info(f"Final average accuracy: {avg_accuracy:.4f}")
            else:
                logger.error("Training failed, skipping evaluation")
                
        elif mode == 'eval':
            # Evaluation only mode
            avg_accuracy, results = run_evaluation_mode(config_dict, args)
            logger.info(f"Evaluation completed. Average accuracy: {avg_accuracy:.4f}")
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        logger.info("Framework execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Exiting gracefully...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Framework execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
