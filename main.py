#!/usr/bin/env python3
"""
Camera Trap Framework V2 - Main Entry Point

A clean, organized, and scalable camera trap evaluation framework.
Supports both config-based and argument-based execution with enhanced logging.

Usage:
    python main.py --camera APN_K024 --config configs/training/baseline.yaml
    python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
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
    from src.data.dataset import get_dataloaders
    DATA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import get_dataloaders: {e}")
    DATA_AVAILABLE = False

# Training modules
try:
    from src.training.oracle import train as train_oracle
    from src.training.accumulative import train as train_accumulative
    from src.training.common import evaluate_checkpoints as eval_per_checkpoint
    from src.training.common import setup_model_and_data as setup_model_and_data_shared
    TRAINING_AVAILABLE = True
except Exception as e:
    logger.warning(f"Training modules not fully available: {e}")
    TRAINING_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Camera Trap Framework V2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--camera', type=str, required=True,
                       help='Camera identifier (e.g., APN_K024)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    
    # Core arguments
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'], 
                       default='train', help='Execution mode')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--train_val', action='store_true', 
                       help='Run validation after each training epoch')
    parser.add_argument('--train_test', action='store_true', 
                       help='Run testing after each training epoch')
    
    # Model arguments  
    parser.add_argument('--model_version', type=str, choices=['v1', 'v2'], 
                       default='v2', help='BioCLIP model version')
    parser.add_argument('--use_peft', action='store_true', 
                       help='Use parameter-efficient fine-tuning')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    
    # System arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--timestamps', action='store_true', 
                       help='Enable timestamps in console logs')
    parser.add_argument('--gpu_cleanup', action='store_true', 
                       help='Enable GPU memory cleanup')
    parser.add_argument('--no_save', action='store_true', 
                       help='Skip saving results')
    
    # Evaluation arguments
    parser.add_argument('--eval_only', action='store_true', 
                       help='Only run evaluation')
    parser.add_argument('--calibration', action='store_true', 
                       help='Enable calibration')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    # Legacy module arguments (for compatibility)
    parser.add_argument('--al_method', type=str, default='all', 
                       help='Active learning method')
    parser.add_argument('--ood_method', type=str, default='all', 
                       help='OOD detection method')
    parser.add_argument('--cl_method', type=str, default='naive-ft', 
                       help='Continual learning method')
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment environment and configuration."""
    # Set up logging first
    icicle_logger.setup_enhanced_logging(use_timestamps=args.timestamps)
    
    # Determine mode type
    mode_type = get_mode_type(args.config)
    
    # Setup experiment directories
    log_dir, timestamp = setup_experiment_directories(args.camera, mode_type)
    
    # Setup proper logging with file handler
    logger_instance = icicle_logger.setup_logging(
        output_dir=log_dir, 
        debug=args.debug, 
        experiment_name="log",
        use_timestamps=args.timestamps
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup GPU manager if needed
    gpu_manager = GPUManager(enable_cleanup=args.gpu_cleanup)
    
    # Validate camera data (silent validation)
    if not validate_camera_data(args.camera):
        logger.error(f"Camera data validation failed for {args.camera}")
        sys.exit(1)
    
    return log_dir, timestamp, mode_type, gpu_manager


def load_and_validate_config(args):
    """Load and validate configuration."""
    # Create config manager with file path 
    config = ConfigManager(args.config)
    
    # Get the loaded configuration dictionary
    config_dict = config.get_config()
    
    # Update with command line arguments
    config_dict = update_config_with_args(config_dict, args)
    
    # Add camera-specific data paths
    camera_name = args.camera
    # Derive project/dataset name from the camera prefix (e.g., MAD_A05 -> MAD)
    project_name = camera_name.split('_', 1)[0] if '_' in camera_name else camera_name
    camera_data_dir = f"data/{project_name}/{camera_name}/30"
    
    # Ensure data section exists
    if 'data' not in config_dict:
        config_dict['data'] = {}
    
    # Set camera-specific data paths
    config_dict['data']['camera'] = camera_name
    config_dict['data']['data_dir'] = camera_data_dir
    config_dict['data']['train_path'] = f"{camera_data_dir}/train.json"
    config_dict['data']['test_path'] = f"{camera_data_dir}/test.json"
    config_dict['data']['train_all_path'] = f"{camera_data_dir}/train-all.json"
    
    # Validate configuration (silent validation)
    if not validate_config(config_dict):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    return config, config_dict


def setup_model_and_data(config, args, mode='oracle', current_checkpoint=None):
    """Backward-compat wrapper to shared setup."""
    return setup_model_and_data_shared(config, args, mode=mode, current_checkpoint=current_checkpoint)


def evaluate_model_checkpoint_based(config, args, trained_model=None):
    """Delegate to training.common.evaluate_checkpoints for real evaluation."""
    return eval_per_checkpoint(config, args, trained_model)


def run_training_mode(config, args, mode_type):
    """Run training via dedicated modules based on mode type."""
    # Read epochs from dict
    training_epochs = (config.get('training', {}) or {}).get('epochs', 30)
    if training_epochs == 0 or mode_type == 'zs':
        return None, 'zero_shot'
    if mode_type == 'accumulative':
        return train_accumulative(config, args), 'accumulative'
    if mode_type == 'oracle':
        return train_oracle(config, args), 'oracle'
    return train_oracle(config, args), 'default'


def load_checkpoint_data(data_path):
    """Kept for compatibility. Prefer src.utils.paths.load_checkpoint_data."""
    import json
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def create_checkpoint_dataset(checkpoint_samples, class_names):
    """Deprecated: use training.common helpers when needed."""
    from src.training.common import _build_simple_dataset
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return _build_simple_dataset(checkpoint_samples, class_to_idx)


def calculate_balanced_accuracy(predictions, labels, num_classes):
    """Deprecated: use MetricsCalculator from src.utils.metrics."""
    import numpy as np
    from src.utils.metrics import MetricsCalculator
    predictions = np.array(predictions)
    labels = np.array(labels)
    mc = MetricsCalculator([str(i) for i in range(num_classes)])
    m = mc.calculate_metrics(predictions, labels)
    return float(m['balanced_accuracy'])


def evaluate_epoch(model, data_loader, criterion, device, mode_type="oracle"):
    """Deprecated shim to training.common.evaluate_epoch."""
    from src.training.common import evaluate_epoch as _eval
    return _eval(model, data_loader, criterion, device)


def evaluate_oracle_per_checkpoint(model, config, criterion, device):
    """Deprecated shim to module in src.training.oracle."""
    from src.training.oracle import evaluate_oracle_per_checkpoint as _eval
    return _eval(model, config, criterion, device)


def train_model_oracle(config, args):
    """Delegates to src.training.oracle.train"""
    return train_oracle(config, args)
    
    # Create dataset and dataloader
    train_dataset = create_checkpoint_dataset(all_samples, class_names)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('training.batch_size', 32),
        shuffle=True,
        num_workers=config.get('training.num_workers', 4)
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate', 0.0001),
        weight_decay=config.get('training.weight_decay', 0.0001)
    )
    
    # Use command line epochs if provided
    num_epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else config.get('training.epochs', 30)
    
    # Log training completion
    icicle_logger.log_training_completion("oracle")
    
    # Save model
    model_path = 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    icicle_logger.log_model_info(f"Model saved to {model_path}")    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total if total > 0 else 0
        lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results with clean format
        icicle_logger.log_training_epoch(
            epoch=epoch, 
            phase="TRAIN", 
            loss=train_loss, 
            acc=train_acc/100, 
            bal_acc=train_acc/100,  # Using same as acc for simplicity
            lr=lr,
            samples=total
        )
        
        # Add separator between epochs
        if epoch < num_epochs - 1:
            icicle_logger.log_training_separator()
    
    # Log training completion
    icicle_logger.log_training_completion("oracle")
    
    # Save model
    model_path = 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    icicle_logger.log_model_info(f"Model saved to {model_path}")
    
    return model


def train_model_accumulative(config, args):
    """Delegates to src.training.accumulative.train"""
    return train_accumulative(config, args)


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup experiment
        log_dir, timestamp, mode_type, gpu_manager = setup_experiment(args)
        
        # Load and validate configuration
        config, config_dict = load_and_validate_config(args)
        
        # Add the original config file path to config_dict for logging
        config_dict['config'] = args.config
        
        # Get checkpoint information for initial setup
        checkpoints = get_checkpoint_directories(args.camera)
        train_path = config_dict['data']['train_path']
        test_path = config_dict['data']['test_path']
        
        # ========== PHASE 1: SETUP DETAILS (merged with initial setup) ==========
        model_info = {
            'name': 'BioCLIP',
            'source': 'loaded from pre-trained (original)',
            'num_classes': 'auto-detected'
        }
        
        icicle_logger.log_setup_details(
            camera=args.camera,
            log_location=log_dir,
            model_info=model_info,
            config_dict=config_dict,
            num_checkpoints=len(checkpoints),
            train_path=train_path,
            test_path=test_path
        )
        
        # ========== PHASE 2: DATASET PREPARATION ==========
        # Load and analyze data to get dataset details
        
        # Load checkpoint data function
        def load_checkpoint_data_local(data_path):
            import json
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        
        train_data = load_checkpoint_data_local(config_dict['data']['train_path'])
        test_data = load_checkpoint_data_local(config_dict['data']['test_path'])
        
        # Extract class information for dataset overview
        all_classes = set()
        train_samples_per_class = {}
        test_samples_per_class = {}
        total_train_samples = 0
        total_test_samples = 0
        
        for ckp_key, samples in train_data.items():
            if ckp_key.startswith('ckp_'):
                for sample in samples:
                    class_name = sample['common']
                    all_classes.add(class_name)
                    if class_name not in train_samples_per_class:
                        train_samples_per_class[class_name] = 0
                    train_samples_per_class[class_name] += 1
                    total_train_samples += 1
        
        # Calculate test samples per class
        for ckp_key, samples in test_data.items():
            if ckp_key.startswith('ckp_'):
                for sample in samples:
                    class_name = sample['common']
                    all_classes.add(class_name)
                    if class_name not in test_samples_per_class:
                        test_samples_per_class[class_name] = 0
                    test_samples_per_class[class_name] += 1
                    total_test_samples += 1
        
        # Build simple class distribution for overview (Train/Test only)
        class_distribution_overview = {}
        for class_name in sorted(all_classes):
            train_count = train_samples_per_class.get(class_name, 0)
            test_count = test_samples_per_class.get(class_name, 0)
            class_distribution_overview[class_name] = {
                'train': train_count,
                'test': test_count
            }
        
        # Get checkpoint information
        train_checkpoints = [key for key in train_data.keys() if key.startswith('ckp_')]
        test_checkpoints = [key for key in test_data.keys() if key.startswith('ckp_')]
        num_checkpoints = len(test_checkpoints)
        
        icicle_logger.log_dataset_details(
            train_size=total_train_samples,
            test_size=total_test_samples,
            num_checkpoints=num_checkpoints,
            num_classes=len(all_classes),
            class_distribution=class_distribution_overview,
            checkpoint_list=test_checkpoints
        )
        
        # Update config with detected num_classes
        config_dict['model']['num_classes'] = len(all_classes)
        
        # ========== PHASE 3: TRAINING PHASE ==========
        if args.mode != 'eval' and not args.eval_only:
            trained_model, training_type = run_training_mode(config_dict, args, mode_type)
        else:
            trained_model = None
        
        # ========== PHASE 4: TESTING PHASE ==========
        icicle_logger.log_testing_phase_header(num_checkpoints)
        
        # Initialize results manager
        results_manager = ResultsManager(log_dir)
        results_manager.set_experiment_info(args.camera, args.mode, config_dict)
        
        # Run evaluation
        accuracy, checkpoint_results = evaluate_model_checkpoint_based(config_dict, args, trained_model)
        
        # Store results
        for checkpoint, result in checkpoint_results.items():
            results_manager.add_checkpoint_result(
                checkpoint=checkpoint,
                metrics=result['metrics'],
                sample_count=result['sample_count']
            )
        
        # Calculate and save summary
        results_manager.calculate_summary()
        results_file = results_manager.save_results()
        
        # ========== PHASE 5: FINAL SUMMARY ==========
        # Prepare summary data
        summary_data = {
            'num_checkpoints': len(checkpoint_results),
            'average_accuracy': sum(r['metrics']['accuracy'] for r in checkpoint_results.values()) / len(checkpoint_results) if checkpoint_results else 0.0,
            'average_balanced_accuracy': sum(r['metrics']['balanced_accuracy'] for r in checkpoint_results.values()) / len(checkpoint_results) if checkpoint_results else 0.0
        }
        
        # Add best/worst checkpoint info
        if checkpoint_results:
            best_ckp = max(checkpoint_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
            worst_ckp = min(checkpoint_results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])
            
            summary_data['best_checkpoint'] = best_ckp[0]
            summary_data['best_accuracy'] = best_ckp[1]['metrics']['balanced_accuracy']
            summary_data['worst_checkpoint'] = worst_ckp[0]
            summary_data['worst_accuracy'] = worst_ckp[1]['metrics']['balanced_accuracy']
        
        icicle_logger.log_final_summary(summary_data)
        
    except Exception as e:
        logger.error(f"Framework execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # GPU cleanup
        if 'gpu_manager' in locals():
            gpu_manager.cleanup()


if __name__ == '__main__':
    main()
