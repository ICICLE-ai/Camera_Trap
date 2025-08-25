"""
Common training utilities shared between different training modes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import contextlib
import sys
import io

logger = logging.getLogger(__name__)


def setup_training_device(args):
    """Setup training device (CPU/GPU)."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    return device


def extract_class_names(train_data, test_data):
    """Extract unique class names from training and test data."""
    all_classes = set()
    
    # Extract from training data
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])
    
    # Extract from test data
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])
    
    return sorted(list(all_classes))


def validate_data_paths(config):
    """Validate that training and test data paths exist."""
    data_config = config.get('data', {})
    train_path = data_config.get('train_path')
    test_path = data_config.get('test_path')
    
    if not train_path or not Path(train_path).exists():
        logger.error(f"Training data path not found: {train_path}")
        return False, None, None
        
    if not test_path or not Path(test_path).exists():
        logger.error(f"Test data path not found: {test_path}")
        return False, None, None
    
    return True, train_path, test_path


def setup_training_config(config, num_classes):
    """Update config with detected number of classes."""
    if 'model' not in config:
        config['model'] = {}
    config['model']['num_classes'] = num_classes
    return config


def get_training_hyperparameters(config, args):
    """Extract training hyperparameters from config and args."""
    num_epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else config.get('training.epochs', 30)
    learning_rate = config.get('training.learning_rate', 0.0001)
    weight_decay = config.get('training.weight_decay', 0.0001)
    
    return {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }


def create_optimizer_and_criterion(model, hyperparams):
    """Create optimizer and loss criterion."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )
    return optimizer, criterion


def log_epoch_results(epoch, phase, loss, acc, bal_acc, lr=None, samples=None, indent=""):
    """Log epoch results in consistent format."""
    if phase == "TRAIN":
        emoji = "🔹"
    elif "VAL" in phase:
        emoji = "🔸"
    elif "TEST" in phase:
        emoji = "🔻"
    else:
        emoji = "📊"
    
    log_line = f"{indent}{emoji} Epoch {epoch:2d} [{phase}] Loss: {loss:.4f} | Acc: {acc:.4f} | Bal.Acc: {bal_acc:.4f}"
    
    if lr is not None:
        log_line += f" | LR: {lr:.8f}"
    
    if samples is not None:
        log_line += f" | Samples: {samples}"
    
    print(log_line)


@contextlib.contextmanager
def suppress_verbose_logs():
    """Context manager to suppress verbose model creation logs."""
    old_level = logging.getLogger().level
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        logging.getLogger().setLevel(logging.ERROR)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        logging.getLogger().setLevel(old_level)
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def extract_checkpoint_names(data, prefix='ckp_'):
    """Extract and sort checkpoint names from data."""
    checkpoints = [key for key in data.keys() if key.startswith(prefix)]
    checkpoints.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
    return checkpoints
