"""
Oracle training implementation for Camera Trap Framework V2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


def train_model_oracle(config, args):
    """Train the model in oracle mode - training on all available data with smart validation."""
    logger.info("Starting oracle training...")
    logger.info(f"Using device: {args.device}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load training and test data for Oracle validation strategy
    data_config = config.get('data', {})
    train_path = data_config.get('train_path')
    test_path = data_config.get('test_path')
    
    if not train_path or not Path(train_path).exists():
        logger.error(f"Training data path not found: {train_path}")
        return None
        
    if not test_path or not Path(test_path).exists():
        logger.error(f"Test data path not found: {test_path}")
        return None
    
    # Load checkpoint data
    train_data = load_checkpoint_data(train_path)
    test_data = load_checkpoint_data(test_path)
    
    # Extract class names from both training and test data
    all_classes = set()
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])
    
    class_names = sorted(list(all_classes))
    num_classes = len(class_names)
    
    logger.info(f"Found {num_classes} classes: {class_names}")
    
    # Update config with detected num_classes
    if 'model' not in config:
        config['model'] = {}
    config['model']['num_classes'] = num_classes
    
    # Create model
    try:
        from ..models.factory import create_model
        model = create_model(config)
    except ImportError as e:
        logger.error(f"Could not import model factory: {e}")
        logger.info("Creating simple model placeholder...")
        # Create a simple dummy model for testing
        from torchvision import models
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)
    
    # Get train and validation data using Oracle validation strategy (2 samples per class for validation)
    train_loader, val_loader = create_oracle_data_loaders(train_data, class_names, config)
    
    if train_loader is None:
        logger.error("Failed to create training data loader")
        return None
    
    # Calculate and log per-class distribution for Oracle mode
    oracle_class_distribution = calculate_oracle_class_distribution(train_data, test_data, val_loader)
    
    # Oracle training phase header
    print("================================================================================")
    print("🏋️  TRAINING PHASE")
    print("================================================================================")
    print("")
    print("🚀 Training Information:")
    print("   Parameter        : Value          ")
    print("   ---------------- : ---------------")
    print(f"   Mode             : oracle training")
    print(f"   Epochs           : {args.epochs if hasattr(args, 'epochs') and args.epochs else config.get('training', {}).get('epochs', 30)}")
    print(f"   Classes          : {num_classes}")
    print(f"   Val Samples      : {len(val_loader.dataset) if val_loader else 0}")
    
    if train_loader:
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset) if val_loader else 0
        print(f"   Total Samples    : {train_samples + val_samples}")
    else:
        print(f"   Total Samples    : Unknown")
    print("")
    
    # Log the Oracle per-class distribution
    try:
        from ..utils.logging import ICICLELogger
        icicle_logger = ICICLELogger()
        icicle_logger.log_oracle_class_distribution(oracle_class_distribution)
    except Exception as e:
        logger.warning(f"Could not log class distribution: {e}")
    
    # Training setup
    num_epochs = args.epochs if hasattr(args, 'epochs') and args.epochs else config.get('training', {}).get('epochs', 30)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training', {}).get('optimizer_params', {}).get('lr', 0.0001),
        weight_decay=config.get('training', {}).get('optimizer_params', {}).get('weight_decay', 0.0001)
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        try:
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
        except Exception as e:
            logger.error(f"Training error at epoch {epoch}: {e}")
            logger.warning("Training data might not be properly loaded")
            return None
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        
        # Log epoch results with clean format (matching accumulative style)
        print(f"🔹 Epoch {epoch:2d} [TRAIN] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal.Acc: {train_acc:.4f}")
        
        # Run validation if requested
        if args.train_val and val_loader and len(val_loader) > 0:
            try:
                val_loss, val_acc, val_bal_acc, val_samples = evaluate_epoch(
                    model, val_loader, criterion, device, mode_type="oracle"
                )
                print(f"🔸 Epoch {epoch:2d} [ VAL ] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Bal.Acc: {val_bal_acc:.4f}")
            except Exception as e:
                logger.warning(f"Validation error: {e}")
        
        # Run testing if requested - Oracle tests on EACH checkpoint and averages
        if args.train_test:
            try:
                test_loss, test_acc, test_bal_acc, test_samples = evaluate_oracle_per_checkpoint(
                    model, test_data, criterion, device, class_names
                )
                print(f"🔻 Epoch {epoch:2d} [TEST*] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Bal.Acc: {test_bal_acc:.4f}")
            except Exception as e:
                logger.warning(f"Testing error: {e}")
    
    # Log training completion
    try:
        from ..utils.logging import ICICLELogger
        icicle_logger = ICICLELogger()
        icicle_logger.log_training_completion("oracle")
    except Exception as e:
        logger.warning(f"Could not log training completion: {e}")
    
    # Save model
    model_path = 'oracle_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Oracle training completed!")
    return model


def load_checkpoint_data(data_path):
    """Load checkpoint data from JSON file."""
    try:
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load checkpoint data from {data_path}: {e}")
        return {}


def create_oracle_data_loaders(train_data, class_names, config):
    """Create data loaders for Oracle training using validation strategy."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import random
    
    # Create class mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect all training samples
    all_train_samples = []
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            all_train_samples.extend(samples)
    
    if not all_train_samples:
        logger.error("No training samples found")
        return None, None
    
    # Separate samples by class for validation split
    class_samples = {}
    for sample in all_train_samples:
        class_name = sample['common']
        if class_name not in class_samples:
            class_samples[class_name] = []
        class_samples[class_name].append(sample)
    
    # Oracle validation strategy: Use 2 samples per class for validation
    train_samples = []
    val_samples = []
    
    for class_name, samples in class_samples.items():
        if len(samples) >= 3:  # Need at least 3 samples to split
            random.shuffle(samples)
            val_samples.extend(samples[:2])  # First 2 for validation
            train_samples.extend(samples[2:])  # Rest for training
        else:
            # If less than 3 samples, put all in training
            train_samples.extend(samples)
    
    logger.info(f"Oracle split: {len(train_samples)} train, {len(val_samples)} validation samples")
    
    # Create datasets
    class OracleDataset(Dataset):
        def __init__(self, samples, class_to_idx, transform=None):
            self.samples = samples
            self.class_to_idx = class_to_idx
            self.transform = transform
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            image_path = sample['image_path']
            class_name = sample['common']
            
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                label = self.class_to_idx[class_name]
                
                return {
                    'image': image,
                    'label': label,
                    'common_name': class_name
                }
            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}")
                # Return a dummy sample
                dummy_image = torch.zeros(3, 224, 224)
                return {
                    'image': dummy_image,
                    'label': 0,
                    'common_name': class_names[0]
                }
    
    # Create data loaders
    train_dataset = OracleDataset(train_samples, class_to_idx, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )
    
    val_loader = None
    if val_samples:
        val_dataset = OracleDataset(val_samples, class_to_idx, transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('training', {}).get('batch_size', 32),
            shuffle=False,
            num_workers=4
        )
    
    return train_loader, val_loader


def calculate_oracle_class_distribution(train_data, test_data, val_loader):
    """Calculate per-class distribution for Oracle mode."""
    oracle_class_distribution = {}
    
    # Count original training samples per class (before validation split)
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                class_name = sample['common']
                if class_name not in oracle_class_distribution:
                    oracle_class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
                oracle_class_distribution[class_name]['train'] += 1
    
    # Count validation samples (these were taken from training)
    if val_loader:
        try:
            for batch in val_loader:
                for common_name in batch['common_name']:
                    if common_name not in oracle_class_distribution:
                        oracle_class_distribution[common_name] = {'train': 0, 'val': 0, 'test': 0}
                    oracle_class_distribution[common_name]['val'] += 1
                    # Subtract from training count since these came from training
                    oracle_class_distribution[common_name]['train'] -= 1
        except Exception as e:
            logger.warning(f"Could not count validation samples: {e}")
    
    # Count test samples
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                class_name = sample['common']
                if class_name not in oracle_class_distribution:
                    oracle_class_distribution[class_name] = {'train': 0, 'val': 0, 'test': 0}
                oracle_class_distribution[class_name]['test'] += 1
    
    return oracle_class_distribution


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


def evaluate_oracle_per_checkpoint(model, test_data, criterion, device, class_names):
    """Evaluate Oracle model across all test checkpoints and return averaged results."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Create class mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
            
            # Create test dataset for this checkpoint
            class TestDataset(Dataset):
                def __init__(self, samples, class_to_idx, transform=None):
                    self.samples = samples
                    self.class_to_idx = class_to_idx
                    self.transform = transform
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    sample = self.samples[idx]
                    image_path = sample['image_path']
                    class_name = sample['common']
                    
                    try:
                        image = Image.open(image_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        
                        label = self.class_to_idx[class_name]
                        
                        return {
                            'image': image,
                            'label': label,
                            'common_name': class_name
                        }
                    except Exception as e:
                        logger.warning(f"Error loading image {image_path}: {e}")
                        # Return a dummy sample
                        dummy_image = torch.zeros(3, 224, 224)
                        return {
                            'image': dummy_image,
                            'label': 0,
                            'common_name': class_names[0]
                        }
            
            test_dataset = TestDataset(test_data[checkpoint], class_to_idx, transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
            
            if len(test_loader) == 0:
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
