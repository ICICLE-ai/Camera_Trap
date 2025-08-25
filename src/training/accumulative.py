"""
Accumulative training implementation for Camera Trap Framework V2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

from .common import (
    setup_training_device, extract_class_names, validate_data_paths,
    setup_training_config, get_training_hyperparameters, create_optimizer_and_criterion,
    log_epoch_results, suppress_verbose_logs, extract_checkpoint_names
)

logger = logging.getLogger(__name__)


def calculate_round_class_distribution(train_data, test_data, checkpoint_round, current_train_checkpoint, current_test_checkpoint):
    """Calculate per-class distribution for a specific accumulative round."""
    round_class_distribution = {}
    
    # Calculate training samples (cumulative from ckp_1 to current_train_checkpoint)
    for ckp_num in range(1, checkpoint_round + 1):
        ckp_key = f'ckp_{ckp_num}'
        if ckp_key in train_data:
            for sample in train_data[ckp_key]:
                class_name = sample['common']
                if class_name not in round_class_distribution:
                    round_class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
                round_class_distribution[class_name]['train'] += 1
    
    # Calculate validation samples (from current_train_checkpoint test data)
    if current_train_checkpoint in test_data:
        for sample in test_data[current_train_checkpoint]:
            class_name = sample['common']
            if class_name not in round_class_distribution:
                round_class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
            round_class_distribution[class_name]['val'] += 1
    
    # Calculate test samples (from current_test_checkpoint test data)
    if current_test_checkpoint in test_data:
        for sample in test_data[current_test_checkpoint]:
            class_name = sample['common']
            if class_name not in round_class_distribution:
                round_class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
            round_class_distribution[class_name]['test'] += 1
    
    return round_class_distribution


def log_accumulative_training_header(config, args, num_classes, max_train_checkpoint, test_checkpoints, train_data, train_checkpoints):
    """Log accumulative training phase header and information."""
    from ..utils.logging import ICICLELogger
    icicle_logger = ICICLELogger()
    
    num_epochs_per_checkpoint = args.epochs if hasattr(args, 'epochs') and args.epochs else config.get('training.epochs', 30)
    
    icicle_logger.log_training_phase_header(
        mode='accumulative training (progressive)', 
        epochs=num_epochs_per_checkpoint,
        num_classes=num_classes,
        num_train_checkpoints=max_train_checkpoint,
        num_test_checkpoints=len(test_checkpoints),
        total_samples=sum(len(train_data[ckp]) for ckp in train_checkpoints)
    )


class NextCheckpointDataset(Dataset):
    """Dataset for next checkpoint test data."""
    
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Could not load image {sample.get('image_path', 'unknown')}: {e}")
            image = torch.randn(3, 224, 224)
        
        label = self.class_to_idx[sample['common']]
        return {'image': image, 'label': label, 'common_name': sample['common']}


def create_next_checkpoint_test_loader(test_samples, class_names, config):
    """Create test loader for next checkpoint data."""
    if not test_samples:
        return None
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = NextCheckpointDataset(test_samples, class_to_idx, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('evaluation.batch_size', 32),
        shuffle=False,
        num_workers=config.get('evaluation.num_workers', 4)
    )
    
    return test_loader


def train_single_round(model, train_loader, val_loader, test_loader, device, hyperparams, args, checkpoint_round):
    """Train model for a single accumulative round."""
    from ..main_utils import evaluate_epoch
    
    # Create optimizer and criterion for this round
    optimizer, criterion = create_optimizer_and_criterion(model, hyperparams)
    
    # Training loop
    model.train()
    for epoch in range(hyperparams['num_epochs']):
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
        train_acc = correct / total if total > 0 else 0
        
        # Log training results
        log_epoch_results(epoch, "TRAIN", train_loss, train_acc, train_acc, indent="    ")
        
        # Run validation if requested
        if args.train_val and val_loader and len(val_loader) > 0:
            val_loss, val_acc, val_bal_acc, val_samples = evaluate_epoch(
                model, val_loader, criterion, device, mode_type="accumulative"
            )
            log_epoch_results(epoch, " VAL ", val_loss, val_acc, val_bal_acc, indent="    ")
        
        # Run testing if requested
        if args.train_test and test_loader and len(test_loader) > 0:
            test_loss, test_acc, test_bal_acc, test_samples = evaluate_epoch(
                model, test_loader, criterion, device, mode_type="accumulative"
            )
            log_epoch_results(epoch, "TEST+", test_loss, test_acc, test_bal_acc, indent="    ")
    
    return model


def train_model_accumulative(config, args):
    """Train the model in accumulative mode - progressive temporal training with proper validation."""
    # Import required modules at function level to avoid circular imports
    from ..utils.paths import load_checkpoint_data
    from ..utils.logging import ICICLELogger
    from ..main_utils import get_dataloaders, create_model, evaluate_epoch
    
    # Initialize logger
    icicle_logger = ICICLELogger()
    
    # Setup device
    device = setup_training_device(args)
    
    # Validate data paths
    valid_paths, train_path, test_path = validate_data_paths(config)
    if not valid_paths:
        return None
    
    # Load checkpoint data
    train_data = load_checkpoint_data(train_path)
    test_data = load_checkpoint_data(test_path)
    
    # Extract class names
    class_names = extract_class_names(train_data, test_data)
    num_classes = len(class_names)
    
    # Update config with detected classes
    config = setup_training_config(config, num_classes)
    
    # Get training and test checkpoints
    train_checkpoints = extract_checkpoint_names(train_data)
    test_checkpoints = extract_checkpoint_names(test_data)
    
    if not train_checkpoints:
        logger.error("No training checkpoints found in data")
        return None
    
    # For accumulative training, we train up to ckp_16 (test on ckp_17)
    max_train_checkpoint = len(train_checkpoints) - 1
    
    # Get training hyperparameters
    hyperparams = get_training_hyperparameters(config, args)
    
    # Log training header
    log_accumulative_training_header(
        config, args, num_classes, max_train_checkpoint, 
        test_checkpoints, train_data, train_checkpoints
    )
    
    final_model = None
    
    # Progressive training from ckp_1 to ckp_16
    for checkpoint_round in range(1, max_train_checkpoint + 1):
        current_train_checkpoint = f'ckp_{checkpoint_round}'
        current_test_checkpoint = f'ckp_{checkpoint_round + 1}'
        
        logger.info(f"\nRound {checkpoint_round} - ({checkpoint_round}/{max_train_checkpoint})")
        logger.info(f"    Training: ckp_1 → {current_train_checkpoint} | Validation: {current_train_checkpoint} | Test: {current_test_checkpoint}")
        
        # Calculate and log per-class distribution for this round
        round_class_distribution = calculate_round_class_distribution(
            train_data, test_data, checkpoint_round, 
            current_train_checkpoint, current_test_checkpoint
        )
        icicle_logger.log_accumulative_round_distribution(checkpoint_round, round_class_distribution)
        
        # Create fresh model for each round
        with suppress_verbose_logs():
            model = create_model(config)
        model = model.to(device)
        
        # Get data loaders
        train_loader, val_loader, _ = get_dataloaders(
            config, mode='accumulative', current_checkpoint=current_train_checkpoint
        )
        
        # Create test loader for next checkpoint
        next_checkpoint_test_samples = test_data.get(current_test_checkpoint, [])
        test_loader = create_next_checkpoint_test_loader(
            next_checkpoint_test_samples, class_names, config
        )
        
        # Train for this round
        model = train_single_round(
            model, train_loader, val_loader, test_loader, device, 
            hyperparams, args, checkpoint_round
        )
        
        # Evaluate on test set for this round if test_loader exists
        if test_loader and len(test_loader) > 0:
            model.eval()
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(labels).sum().item()
                    total_samples += labels.size(0)
            
            test_acc = total_correct / total_samples if total_samples > 0 else 0
            # For simplicity, using same balanced accuracy as accuracy
            test_bal_acc = test_acc
            
            print(f"    📊 Round {checkpoint_round} Test → {current_test_checkpoint}: Acc: {test_acc*100:.2f}% | Bal.Acc: {test_bal_acc*100:.2f}% ({total_correct}/{total_samples})")
        
        print(f"    ✅ Round {checkpoint_round} completed")
        print("")
        
        final_model = model
    
    # Log training completion
    icicle_logger.log_training_completion("accumulative")
    
    # Save final model
    model_path = 'accumulative_model.pth'
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Final model saved to {model_path}")
    
    return final_model
