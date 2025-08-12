#!/usr/bin/env python3
"""
ICICLE-Benchmark V2 - Main Entry Point

A clean, organized, and scalable version of the ICICLE benchmark.
Supports both config-based and argument-based execution.

Usage:
    python main.py --camera APN_K024 --config configs/training/baseline.yaml
    python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments with minimal required args."""
    parser = argparse.ArgumentParser(
        description='ICICLE-Benchmark V2: Camera Trap Adaptive Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  python main.py --camera APN_K024 --config configs/training/baseline.yaml
  
  # Using command-line arguments
  python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
        """
    )
    
    # Required arguments
    parser.add_argument('--camera', required=True, 
                       help='Camera/dataset name (e.g., APN_K024)')
    
    # Mode selection
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'eval'],
                       help='Execution mode: train (default), test, or eval')
    
    # Config file option
    parser.add_argument('--config', 
                       help='Path to YAML config file. If provided, overrides CLI args.')
    
    # Core training parameters (used when no config file)
    parser.add_argument('--model', default='bioclip',
                       help='Model name (default: bioclip)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config if specified)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config if specified)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size (overrides config if specified)')
    
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


def set_seed(seed):
    """Set random seed for reproducibility."""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logging(output_dir, debug, experiment_name):
    """Setup logging with proper format."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"{experiment_name}.log")
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    return logging.getLogger(__name__)


def create_config_from_args(args):
    """Create configuration from command line arguments."""
    # If config file provided, load it and override with CLI args
    if args.config:
        from src.config import ConfigManager
        config_manager = ConfigManager(args.config)
        config_dict = config_manager.get_config()
        
        # Override with CLI arguments
        if hasattr(args, 'camera'):
            config_dict['data']['dataset'] = args.camera
            # Update data paths based on camera
            config_dict['data']['data_dir'] = f"data/APN/{args.camera}"
            config_dict['data']['train_path'] = f"data/APN/{args.camera}/30/train.json"
            config_dict['data']['test_path'] = f"data/APN/{args.camera}/30/test.json"
            config_dict['data']['image_dir'] = f"data/APN/{args.camera}/images"
        
        if args.epochs:
            config_dict['training']['epochs'] = args.epochs
        if args.lr:
            config_dict['training']['learning_rate'] = args.lr
        if args.batch_size:
            config_dict['training']['batch_size'] = args.batch_size
        if args.model:
            config_dict['model']['name'] = args.model
        
        # Module configurations
        if hasattr(args, 'ood_method'):
            config_dict['modules']['ood']['method'] = args.ood_method
        if hasattr(args, 'al_method'):
            config_dict['modules']['active_learning']['method'] = args.al_method
        if hasattr(args, 'cl_method'):
            config_dict['modules']['continual_learning']['method'] = args.cl_method
        if hasattr(args, 'use_peft'):
            config_dict['model']['use_peft'] = args.use_peft
        if hasattr(args, 'calibration'):
            config_dict['modules']['calibration']['enabled'] = args.calibration
        
        # Create a simple config object
        class Config:
            def __init__(self, config_dict):
                self.config_dict = config_dict
                self.camera = args.camera
                self.experiment_name = config_dict['experiment']['name']
                self.output_dir = args.output_dir or config_dict['experiment']['output_dir']
                self.debug = args.debug
            
            def get_config(self):
                return self.config_dict
            
            def get(self, key, default=None):
                keys = key.split('.')
                value = self.config_dict
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
        
        return Config(config_dict)
    
    else:
        # Create config from CLI arguments only
        config_dict = {
            'experiment': {
                'name': f'{args.camera}_experiment',
                'output_dir': args.output_dir or f'logs/{args.camera}',
                'device': args.device,
                'seed': args.seed
            },
            'model': {
                'name': args.model,
                'use_peft': args.use_peft,
                'pretrained': True,
                'freeze_backbone': False
            },
            'training': {
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'weight_decay': 0.0001,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR'
            },
            'data': {
                'dataset': args.camera,
                'data_dir': f'data/APN/{args.camera}',
                'train_path': f'data/APN/{args.camera}/30/train.json',
                'test_path': f'data/APN/{args.camera}/30/test.json',
                'image_dir': f'data/APN/{args.camera}/images',
                'image_size': 224,
                'val_split': 0.2
            },
            'modules': {
                'ood': {'method': args.ood_method, 'enabled': True},
                'active_learning': {'method': args.al_method, 'enabled': True},
                'continual_learning': {'method': args.cl_method, 'enabled': True},
                'calibration': {'enabled': args.calibration}
            }
        }
        
        class Config:
            def __init__(self, config_dict):
                self.config_dict = config_dict
                self.camera = args.camera
                self.experiment_name = config_dict['experiment']['name']
                self.output_dir = args.output_dir or config_dict['experiment']['output_dir']
                self.debug = args.debug
            
            def get_config(self):
                return self.config_dict
            
            def get(self, key, default=None):
                keys = key.split('.')
                value = self.config_dict
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
        
        return Config(config_dict)
    """Training function using our tested framework."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    logger.info("Starting training...")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    from src.data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config.get_config())
    
    # Create model
    from src.models.factory import create_model
    model = create_model(config.get_config())
    model = model.to(device)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr or config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay', 0.0001)
    )
    
    # Training parameters
    epochs = args.epochs or config.get('training.epochs')
    if args.mode == 'test':
        epochs = min(3, epochs)  # Limit test runs to 3 epochs
        logger.info(f"Test mode: limiting training to {epochs} epochs")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 50)
        
        # Train
        model.train()
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
            
            if batch_idx % 5 == 0:  # Log every 5 batches
                logger.info(f'  Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc and args.save_model:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"  New best validation accuracy: {best_val_acc:.2f}%")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model


def train_model(config, args):
    """Train the model using the provided configuration."""
    logger.info("Starting training...")
    logger.info(f"Using device: {args.device}")
    
    # Load data
    from src.data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config.get_config()
    )
    
    # Get the actual number of classes from config (updated by get_dataloaders)
    num_classes = config.get('model.num_classes', 11)
    logger.info(f"Using {num_classes} classes for model creation")
    
    # Update config with detected num_classes
    config.update('model.num_classes', num_classes)
    
    # Create model
    from src.models.factory import create_model
    model = create_model(config.get_config())
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('training.learning_rate', 0.0001))
    
    num_epochs = config.get('training.epochs', 30)
    logger.info(f"Training for {num_epochs} epochs")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
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
            
            if batch_idx % 5 == 0:  # Log every 5 batches
                logger.info(f'  Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        logger.info(f'Epoch {epoch+1} completed - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    
    # Save model
    model_path = 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")
    return model


def evaluate_model(config, args):
    """Evaluation function using our tested framework."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    logger.info("Starting evaluation...")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    from src.data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config.get_config())
    
    # Create model
    from src.models.factory import create_model
    model = create_model(config.get_config())
    model = model.to(device)
    
    # Load trained model if available
    model_path = Path('best_model.pth')
    if model_path.exists():
        logger.info("Loading trained model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Trained model loaded successfully!")
    else:
        logger.info("No trained model found, using randomly initialized model")
    
    # Evaluate on test data
    logger.info("Evaluating on test data...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    unique_labels = np.unique(all_labels)
    logger.info(f"Number of classes: {len(unique_labels)}")
    logger.info("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                               target_names=[f"Class_{i}" for i in unique_labels],
                               zero_division=0))
    
    logger.info("Evaluation completed!")
    return accuracy


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("="*80)
        logger.info("CAMERA TRAP FRAMEWORK - STARTING")
        logger.info("="*80)
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Seed: {args.seed}")
        
        # Load configuration
        from src.config import ConfigManager
        config = ConfigManager(args.config)
        
        # Ensure data configuration exists based on camera argument
        if not config.get('data'):
            config.update('data', {})
        
        # Set data paths based on camera argument
        config.update('data.dataset', args.camera)
        config.update('data.data_dir', f'data/APN/{args.camera}')
        config.update('data.train_path', f'data/APN/{args.camera}/30/train.json')
        config.update('data.test_path', f'data/APN/{args.camera}/30/test.json')
        config.update('data.image_dir', f'data/APN/{args.camera}/images')
        config.update('data.image_size', 224)
        config.update('data.val_split', 0.2)
        
        # Apply command line overrides (only if explicitly provided)
        if args.epochs is not None:
            config.update('training.epochs', args.epochs)
        if args.lr is not None:
            config.update('training.learning_rate', args.lr)
        if args.batch_size is not None:
            config.update('training.batch_size', args.batch_size)
        
        # Run based on mode
        if args.mode == 'eval':
            accuracy = evaluate_model(config, args)
            logger.info(f"Final accuracy: {accuracy*100:.2f}%")
            
        elif args.mode in ['train', 'test']:
            # Check if this is zero-shot (no training)
            training_epochs = config.get('training.epochs', 30)
            if training_epochs == 0:
                logger.info("Zero-shot mode detected (epochs=0), skipping training...")
                accuracy = evaluate_model(config, args)
                logger.info(f"Zero-shot accuracy: {accuracy*100:.2f}%")
            else:
                model = train_model(config, args)
                
                # Also run evaluation after training
                logger.info("\nRunning evaluation after training...")
                accuracy = evaluate_model(config, args)
                logger.info(f"Final test accuracy: {accuracy*100:.2f}%")
        
        logger.info("="*80)
        logger.info("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Framework execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
