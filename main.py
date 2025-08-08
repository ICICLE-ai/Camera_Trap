#!/usr/bin/env python3
"""
ICICLE-Benchmark V2 - Main Entry Point

A clean, organized, and scalable version of the ICICLE benchmark.
Supports both config-based and argument-based execution.

Usage:
    python main.py --camera APN_K024 --config configs/experiments/baseline.yaml
    python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import ConfigManager
from src.core.pipeline import Pipeline
from src.utils.logging import setup_logging
from src.utils.gpu import GPUManager
from src.utils.seed import set_seed


def parse_args():
    """Parse command-line arguments with minimal required args."""
    parser = argparse.ArgumentParser(
        description='ICICLE-Benchmark V2: Camera Trap Adaptive Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  python main.py --camera APN_K024 --config configs/experiments/baseline.yaml
  
  # Using command-line arguments
  python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
        """
    )
    
    # Required arguments
    parser.add_argument('--camera', required=True, 
                       help='Camera/dataset name (e.g., APN_K024)')
    
    # Config file option
    parser.add_argument('--config', 
                       help='Path to YAML config file. If provided, overrides CLI args.')
    
    # Core training parameters (used when no config file)
    parser.add_argument('--model', default='bioclip',
                       help='Model name (default: bioclip)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size (default: 128)')
    
    # Module configurations (simplified)
    parser.add_argument('--ood_method', default='all',
                       help='OOD detection method (default: all)')
    parser.add_argument('--al_method', default='all',
                       help='Active learning method (default: all)')
    parser.add_argument('--cl_method', default='naive-ft',
                       help='Continual learning method (default: naive-ft)')
    parser.add_argument('--use_peft', action='store_true',
                       help='Enable parameter-efficient fine-tuning (LoRA)')
    parser.add_argument('--calibration', action='store_true',
                       help='Enable inference-level calibration')
    
    # System options
    parser.add_argument('--device', default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--gpu_cleanup', action='store_true',
                       help='Enable GPU VRAM cleaning (default: disabled)')
    
    # Experiment options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only, no training')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save model checkpoints')
    parser.add_argument('--output_dir', 
                       help='Output directory for logs and checkpoints')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize GPU manager if requested
    gpu_manager = None
    if args.gpu_cleanup:
        gpu_manager = GPUManager()
    
    try:
        # Load and merge configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args)
        
        # Setup logging
        logger = setup_logging(
            output_dir=config.output_dir,
            debug=config.debug,
            experiment_name=f"{config.camera}_{config.experiment_name}"
        )
        
        logger.info("="*80)
        logger.info("ICICLE-Benchmark V2 - Starting Pipeline")
        logger.info("="*80)
        logger.info(f"Configuration: {config}")
        
        # Initialize and run pipeline
        pipeline = Pipeline(config, gpu_manager)
        results = pipeline.run()
        
        logger.info("="*80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Pipeline failed with error: {str(e)}")
            logger.exception("Full traceback:")
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)
    
    finally:
        # Cleanup GPU if manager was initialized
        if gpu_manager:
            gpu_manager.cleanup()


if __name__ == '__main__':
    main()
