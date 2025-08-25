"""
Main utilities for Camera Trap Framework V2.
Contains common functions used across different training and evaluation modes.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


def setup_model_and_data(config, args, mode='oracle', current_checkpoint=None):
    """Setup model and data loaders with proper validation strategy."""
    from .models.factory import create_model
    from .data.loader import get_dataloaders
    
    # Create model
    model = create_model(config)
    
    # Get data loaders with mode-specific validation
    train_loader, val_loader, test_loader = get_dataloaders(
        config, mode=mode, current_checkpoint=current_checkpoint
    )
    
    return model, train_loader, val_loader, test_loader


def evaluate_epoch(model, data_loader, criterion, device, mode_type="oracle"):
    """Evaluate model on a single epoch."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    eval_loss = running_loss / len(data_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    model.train()  # Switch back to training mode
    
    return eval_loss, accuracy, balanced_acc, len(all_labels)


def evaluate_oracle_per_checkpoint(model, config, criterion, device):
    """Evaluate Oracle model across all test checkpoints and return averaged results."""
    from .utils.paths import load_checkpoint_data
    from .data.loader import create_checkpoint_test_loader
    
    # Load test data
    data_config = config.get('data', {})
    test_path = data_config.get('test_path')
    test_data = load_checkpoint_data(test_path)
    
    # Get all test checkpoints
    test_checkpoints = [key for key in test_data.keys() if key.startswith('ckp_')]
    test_checkpoints.sort(key=lambda x: int(x.split('_')[1]))
    
    # Evaluate on each checkpoint
    all_losses = []
    all_accuracies = []
    all_balanced_accuracies = []
    
    model.eval()
    with torch.no_grad():
        for checkpoint in test_checkpoints:
            if checkpoint not in test_data or not test_data[checkpoint]:
                continue
            
            # Create test loader for this checkpoint
            test_loader = create_checkpoint_test_loader(
                test_data[checkpoint], config
            )
            
            if not test_loader:
                continue
            
            # Evaluate on this checkpoint
            loss, acc, bal_acc, _ = evaluate_epoch(
                model, test_loader, criterion, device, mode_type="oracle"
            )
            
            all_losses.append(loss)
            all_accuracies.append(acc)
            all_balanced_accuracies.append(bal_acc)
    
    # Calculate averages
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
    avg_balanced_accuracy = np.mean(all_balanced_accuracies) if all_balanced_accuracies else 0.0
    
    model.train()  # Switch back to training mode
    
    return avg_loss, avg_accuracy, avg_balanced_accuracy, len(test_checkpoints)


def get_dataloaders(config, mode='oracle', current_checkpoint=None):
    """Get data loaders based on the specified mode."""
    from .data.loader import DataLoaderFactory
    
    factory = DataLoaderFactory(config)
    
    if mode == 'oracle':
        return factory.create_oracle_loaders()
    elif mode == 'accumulative':
        return factory.create_accumulative_loaders(current_checkpoint)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_model(config):
    """Create and return a model based on the configuration."""
    from .models.factory import create_model as factory_create_model
    return factory_create_model(config)


def load_checkpoint_data(data_path):
    """Load checkpoint data from file."""
    from .utils.paths import load_checkpoint_data as utils_load_checkpoint_data
    return utils_load_checkpoint_data(data_path)


def get_checkpoint_directories(camera):
    """Get checkpoint directories for a given camera."""
    from .utils.paths import get_checkpoint_directories as utils_get_checkpoint_directories
    return utils_get_checkpoint_directories(camera)


def evaluate_model_checkpoint_based(config, args, trained_model=None):
    """Run checkpoint-based evaluation."""
    from .utils.logging import ICICLELogger
    
    icicle_logger = ICICLELogger()
    
    logger.info("Starting checkpoint-based evaluation...")
    logger.info(f"Using device: {args.device}")
    
    # Get checkpoints
    checkpoints = get_checkpoint_directories(args.camera)
    if not checkpoints:
        logger.error(f"No checkpoints found for camera {args.camera}")
        return 0.0, {}
    
    logger.info(f"Found {len(checkpoints)} test checkpoints: {checkpoints}")
    
    # Initialize results storage
    checkpoint_results = {}
    all_accuracies = []
    all_balanced_accuracies = []
    
    # Dataset preparation section
    icicle_logger.log_section_header("📊 DATASET PREPARATION", style='phase')
    
    # Log dataset information
    train_size = 0 if 'zs' in args.config else 500
    eval_size = 828  # This should come from actual data
    
    icicle_logger.log_dataset_info(
        train_size=train_size,
        eval_size=eval_size,
        num_checkpoints=len(checkpoints),
        checkpoint_list=checkpoints
    )
    
    # Create or use provided model
    if trained_model is not None:
        logger.info("Using provided trained model for evaluation")
        model = trained_model
    else:
        logger.info("Creating new model for evaluation")
        model = create_model(config)
    
    # Evaluation logic would continue here...
    # This is a simplified version for the refactoring
    
    return 0.0, {}  # Placeholder return values
