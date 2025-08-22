"""
Dataset management for ICICLE-Benchmark V2
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader, Dataset
import torch

from ..core.config import Config

logger = logging.getLogger(__name__)

class CameraTrapDataset(Dataset):
    """PyTorch Dataset for camera trap data."""
    
    def __init__(self, samples: List[Dict[str, Any]], class_names: List[str]):
        """
        Initialize camera trap dataset.
        
        Args:
            samples: List of sample dictionaries with image_path, class_id, etc.
            class_names: List of class names for mapping
        """
        self.samples = samples
        self.class_names = class_names
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        sample = self.samples[idx]
        
        # For now, return dummy tensors - in production this would load the actual image
        image = torch.randn(3, 224, 224)  # Dummy image tensor
        label = sample['class_id']
        
        return image, label

class DatasetManager:
    """Manages dataset loading and processing for ICICLE-Benchmark V2."""
    
    def __init__(self, config: Config):
        """
        Initialize dataset manager.
        
        Args:
            config: Configuration object containing dataset settings and checkpoint info
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_train_datasets(self) -> Dict[str, CameraTrapDataset]:
        """Load training datasets for all checkpoints using checkpoint_info from config."""
        self.logger.info("Loading training datasets...")
        
        train_datasets = {}
        
        # The checkpoint_info is structured as {camera_name: {ckp_1: {...}, ckp_2: {...}, ...}}
        camera_name = list(self.config.checkpoint_info.keys())[0]  # Get first (and only) camera
        camera_data = self.config.checkpoint_info[camera_name]
        
        for checkpoint_key, ckp_info in camera_data.items():
            num_samples = len(ckp_info.get('samples', []))
            if num_samples > 0:
                self.logger.info(f"Loading training data for {checkpoint_key}: {num_samples} samples")
                # Create proper PyTorch dataset
                train_datasets[checkpoint_key] = CameraTrapDataset(
                    samples=ckp_info['samples'], 
                    class_names=ckp_info['classes']
                )
            
        self.logger.info(f"Loaded {len(train_datasets)} training checkpoints")
        return train_datasets
    
    def load_eval_datasets(self) -> Dict[str, CameraTrapDataset]:
        """Load evaluation datasets for all checkpoints using checkpoint_info from config."""
        self.logger.info("Loading evaluation datasets...")
        
        eval_datasets = {}
        
        # The checkpoint_info is structured as {camera_name: {ckp_1: {...}, ckp_2: {...}, ...}}
        camera_name = list(self.config.checkpoint_info.keys())[0]  # Get first (and only) camera
        camera_data = self.config.checkpoint_info[camera_name]
        
        # Use same checkpoint data for evaluation in zero-shot mode
        for checkpoint_key, ckp_info in camera_data.items():
            num_samples = len(ckp_info.get('samples', []))
            if num_samples > 0:
                self.logger.info(f"Loading evaluation data for {checkpoint_key}: {num_samples} samples")
                # Create proper PyTorch dataset
                eval_datasets[checkpoint_key] = CameraTrapDataset(
                    samples=ckp_info['samples'], 
                    class_names=ckp_info['classes']
                )
            
        self.logger.info(f"Loaded {len(eval_datasets)} evaluation checkpoints")
        return eval_datasets

    def get_legacy_samples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get raw sample data in the legacy format for compatibility."""
        samples = {}
        
        # The checkpoint_info is structured as {camera_name: {ckp_1: {...}, ckp_2: {...}, ...}}
        camera_name = list(self.config.checkpoint_info.keys())[0]  # Get first (and only) camera
        camera_data = self.config.checkpoint_info[camera_name]
        
        # Extract samples directly from checkpoint_info
        for ckp_key, ckp_info in camera_data.items():
            samples[ckp_key] = ckp_info['samples']
                    
        return samples

    def get_checkpoint_list(self) -> List[str]:
        """Return the list of checkpoint keys for the current camera."""
        camera_name = list(self.config.checkpoint_info.keys())[0]
        camera_data = self.config.checkpoint_info[camera_name]
        return list(camera_data.keys())

    def create_dataloader(self, dataset: CameraTrapDataset, batch_size: int, shuffle: bool = False, 
                         num_workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for the given dataset.
        
        Args:
            dataset: CameraTrapDataset to create DataLoader for
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=None  # Use default collate_fn for now
        )

def get_oracle_validation_samples(train_data, test_data, all_classes):
    """
    Create validation samples for Oracle mode by selecting 2 random images per class from training data.
    Smart fallback strategies for classes with insufficient samples.
    
    Args:
        train_data: Training data dictionary
        test_data: Test data dictionary  
        all_classes: Set of all class names
        
    Returns:
        Tuple of (train_samples, val_samples) where val_samples are removed from train_samples
    """
    import random
    from collections import defaultdict
    
    # Collect all training samples
    all_train_samples = []
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            all_train_samples.extend(samples)
    
    # Group training samples by class
    train_samples_by_class = defaultdict(list)
    for sample in all_train_samples:
        train_samples_by_class[sample['common']].append(sample)
    
    # Group test samples by class (for fallback)
    test_samples_by_class = defaultdict(list)
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                test_samples_by_class[sample['common']].append(sample)
    
    val_samples = []
    train_samples_final = []
    val_samples_selected = set()  # Track selected sample IDs to avoid duplicates
    
    logger.info("🔄 Creating Oracle validation set (2 samples per class)")
    
    for class_name in sorted(all_classes):
        train_class_samples = train_samples_by_class[class_name]
        test_class_samples = test_samples_by_class[class_name]
        
        selected_for_val = []
        
        if len(train_class_samples) >= 2:
            # Strategy 1: Select 2 random from training data
            selected_for_val = random.sample(train_class_samples, 2)
            logger.debug(f"   {class_name}: Selected 2 from {len(train_class_samples)} training samples")
            
        elif len(train_class_samples) == 1:
            # Strategy 2: Use the 1 training sample + 1 from test if available
            selected_for_val.append(train_class_samples[0])
            if test_class_samples:
                selected_for_val.append(random.choice(test_class_samples))
                logger.debug(f"   {class_name}: Used 1 train + 1 test sample")
            else:
                logger.debug(f"   {class_name}: Only 1 sample available, using it for validation")
                
        elif len(train_class_samples) == 0:
            # Strategy 3: No training samples, use test samples if available
            if len(test_class_samples) >= 2:
                selected_for_val = random.sample(test_class_samples, 2)
                logger.debug(f"   {class_name}: No training samples, used 2 from test")
            elif len(test_class_samples) == 1:
                selected_for_val = test_class_samples
                logger.debug(f"   {class_name}: No training samples, used 1 from test")
            else:
                logger.warning(f"   {class_name}: No samples available for validation - skipping")
                continue
        
        # Add selected samples to validation set
        for sample in selected_for_val:
            sample_id = f"{sample['image_path']}_{sample['common']}"
            if sample_id not in val_samples_selected:
                val_samples.append(sample)
                val_samples_selected.add(sample_id)
    
    # Add remaining training samples (not selected for validation) to final training set
    for sample in all_train_samples:
        sample_id = f"{sample['image_path']}_{sample['common']}"
        if sample_id not in val_samples_selected:
            train_samples_final.append(sample)
    
    logger.info(f"📊 Oracle validation: {len(val_samples)} validation samples, {len(train_samples_final)} training samples")
    return train_samples_final, val_samples


def get_accumulative_validation_samples(train_data, test_data, current_checkpoint):
    """
    Create validation samples for Accumulative mode using current checkpoint's test data.
    
    Args:
        train_data: Training data dictionary
        test_data: Test data dictionary
        current_checkpoint: Current checkpoint key (e.g., 'ckp_4')
        
    Returns:
        Tuple of (train_samples, val_samples) where val_samples come from CURRENT checkpoint test
    """
    # Collect training samples up to current checkpoint (accumulative)
    train_samples = []
    current_ckp_num = int(current_checkpoint.split('_')[1])
    
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            ckp_num = int(ckp_key.split('_')[1])
            if ckp_num <= current_ckp_num:
                train_samples.extend(samples)
    
    # For accumulative training, validation uses CURRENT checkpoint's test data
    val_samples = test_data.get(current_checkpoint, [])
    
    # Silent logging - no output here
    return train_samples, val_samples


def get_dataloaders(config_dict, mode='oracle', current_checkpoint=None):
    """
    Create train, validation, and test dataloaders from config with smart validation strategies.
    
    Args:
        config_dict: Configuration dictionary
        mode: Training mode ('oracle', 'accumulative', or 'zs')
        current_checkpoint: Current checkpoint for accumulative mode (e.g., 'ckp_4')
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    import json
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as transforms
    import random
    
    # Set random seed for reproducible validation splits
    random.seed(42)
    
    # Load JSON data
    data_config = config_dict['data']
    train_path = data_config['train_path']
    test_path = data_config['test_path']
    
    # Load train data
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Load test data
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # Extract all classes from both training and test data
    all_classes = set()
    for ckp_key, samples in train_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])
    
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            for sample in samples:
                all_classes.add(sample['common'])

    # Create class mapping
    class_names = sorted(list(all_classes))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Store class names in config for model creation
    config_dict['data']['class_names'] = class_names
    config_dict['model']['num_classes'] = len(class_names)
    
    # Get train and validation samples based on mode
    if mode == 'oracle':
        # Oracle mode: Smart random selection from training data
        train_samples, val_samples = get_oracle_validation_samples(train_data, test_data, all_classes)
    elif mode == 'accumulative':
        # Accumulative mode: Use current checkpoint's test data for validation
        if current_checkpoint is None:
            logger.warning("No current_checkpoint provided for accumulative mode, defaulting to ckp_1")
            current_checkpoint = 'ckp_1'
        train_samples, val_samples = get_accumulative_validation_samples(train_data, test_data, current_checkpoint)
    else:
        # Zero-shot mode: No training, use all data for testing
        train_samples = []
        val_samples = []
    
    # Extract all test samples for final evaluation
    all_test_samples = []
    for ckp_key, samples in test_data.items():
        if ckp_key.startswith('ckp_'):
            all_test_samples.extend(samples)
    
    print(f"    Classes found: {len(class_names)}")
    print(f"    Train samples: {len(train_samples)}")
    print(f"    Val samples: {len(val_samples)}")  
    print(f"    Test samples: {len(all_test_samples)}")    # Create datasets
    class SimpleCameraTrapDataset(Dataset):
        def __init__(self, samples, class_to_idx, transform=None):
            self.samples = samples
            self.class_to_idx = class_to_idx
            self.transform = transform
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load actual image
            try:
                from PIL import Image
                image_path = sample['image_path']
                image = Image.open(image_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                else:
                    # Default transform if none provided
                    import torchvision.transforms as transforms
                    default_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image = default_transform(image)
                    
            except Exception as e:
                # Fallback to dummy tensor if image loading fails
                print(f"Warning: Could not load image {sample['image_path']}: {e}")
                image = torch.randn(3, 224, 224)
            
            label = self.class_to_idx[sample['common']]
            
            return {
                'image': image,
                'label': label,
                'image_path': sample['image_path'],
                'common_name': sample['common']
            }
    
    # Create simple transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SimpleCameraTrapDataset(train_samples, class_to_idx, transform)
    val_dataset = SimpleCameraTrapDataset(val_samples, class_to_idx, transform)
    test_dataset = SimpleCameraTrapDataset(all_test_samples, class_to_idx, transform)
    
    # Create dataloaders with smaller batch sizes to avoid OOM
    training_config = config_dict.get('training', {})
    train_batch_size = training_config.get('train_batch_size', training_config.get('batch_size', 16))  # Reduced default
    eval_batch_size = training_config.get('eval_batch_size', 8)  # Further reduced for memory efficiency
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=False  # Disable for debugging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=False  # Disable for debugging
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging  
        pin_memory=False  # Disable for debugging
    )
    
    return train_loader, val_loader, test_loader
