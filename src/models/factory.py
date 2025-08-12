"""
Model factory for ICICLE-Benchmark V2
"""

import os
import logging
from typing import Any
import torch
import torch.nn as nn

from ..core.config import Config

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating different types of models."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_model(self) -> nn.Module:
        """
        Create a model based on configuration.
        
        Returns:
            PyTorch model
        """
        model_name = self.config.model.lower()
        
        logger.info(f"Creating model: {model_name}")
        
        if model_name == 'bioclip':
            model = self._create_bioclip_model()
        elif model_name in ['resnet50', 'resnet']:
            model = self._create_resnet_model()
        elif model_name.startswith('vit'):
            model = self._create_vit_model()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Apply PEFT if enabled
        if self.config.use_peft:
            model = self._apply_peft(model)
            
        return model
    
    def _create_bioclip_model(self) -> nn.Module:
        """Create BioCLIP model."""
        try:
            # This would use the BioCLIP implementation from the original code
            # For now, return a placeholder model
            logger.warning("BioCLIP model not yet implemented, using placeholder")
            return PlaceholderModel(len(self.config.class_names))
            
        except Exception as e:
            logger.error(f"Failed to create BioCLIP model: {e}")
            raise
    
    def _create_resnet_model(self) -> nn.Module:
        """Create ResNet model."""
        try:
            import torchvision.models as models
            
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(self.config.class_names))
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create ResNet model: {e}")
            return PlaceholderModel(len(self.config.class_names))
    
    def _create_vit_model(self) -> nn.Module:
        """Create Vision Transformer model."""
        try:
            # This would implement ViT model creation
            logger.warning("ViT model not yet implemented, using placeholder")
            return PlaceholderModel(len(self.config.class_names))
            
        except Exception as e:
            logger.error(f"Failed to create ViT model: {e}")
            return PlaceholderModel(len(self.config.class_names))
    
    def _apply_peft(self, model: nn.Module) -> nn.Module:
        """Apply Parameter-Efficient Fine-Tuning (LoRA)."""
        try:
            logger.info("Applying PEFT (LoRA) to model")
            
            # This would implement LoRA or other PEFT methods
            # For now, just return the original model
            logger.warning("PEFT not yet implemented")
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply PEFT: {e}")
            return model


class PlaceholderModel(nn.Module):
    """
    Placeholder model for testing purposes.
    Simple CNN that can be used when specific models are not available.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_feats=False):
        """Forward pass with optional feature return."""
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        
        # Get features before final classification
        intermediate_features = self.classifier[:-1](features_flat)
        logits = self.classifier[-1](intermediate_features)
        
        if return_feats:
            return logits, intermediate_features
        return logits


def create_model(config_dict):
    """
    Create a model based on configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary containing model settings
        
    Returns:
        PyTorch model
    """
    import logging
    logger = logging.getLogger(__name__)
    
    model_config = config_dict['model']
    model_name = model_config['name'].lower()
    num_classes = model_config.get('num_classes', 11)  # Auto-detected or default
    
    logger.info(f"Creating model: {model_name}")
    
    if model_name == 'bioclip':
        # Use BioCLIP model with pre-trained weights
        from .bioclip_model import create_bioclip_model
        
        # Generate class names (in real scenario, these come from data)
        class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Try to get actual class names from data config if available
        data_config = config_dict.get('data', {})
        if 'class_names' in data_config and data_config['class_names']:
            class_names = data_config['class_names']
        
        # Look for pre-trained weights
        pretrained_path = None
        bioclip_weight_paths = [
            'pretrained_weight/bioclip/open_clip_pytorch_model.bin',
            'ICICLE-Benchmark/pretrained_weight/bioclip/open_clip_pytorch_model.bin'
        ]
        
        for path in bioclip_weight_paths:
            if os.path.exists(path):
                pretrained_path = path
                logger.info(f"Found BioCLIP weights at: {path}")
                break
        
        if pretrained_path is None:
            logger.warning("No local BioCLIP weights found in expected locations")
            logger.info("Expected locations:")
            for path in bioclip_weight_paths:
                logger.info(f"  - {path}")
            logger.info("Falling back to PlaceholderModel")
            model = PlaceholderModel(num_classes)
        else:
            try:
                model = create_bioclip_model(
                    num_classes=num_classes,
                    class_names=class_names,
                    pretrained_path=pretrained_path,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                logger.info(f"Created BioCLIP model with {num_classes} classes")
                
            except Exception as e:
                logger.error(f"Failed to create BioCLIP model: {e}")
                logger.info("Falling back to PlaceholderModel")
                model = PlaceholderModel(num_classes)
            
    elif model_name == 'resnet50':
        # Use ResNet50 with ImageNet pre-training
        try:
            import torchvision.models as models
            model = models.resnet50(weights='IMAGENET1K_V2')
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            logger.info(f"Created ResNet50 with {num_classes} classes")
            
        except Exception as e:
            logger.error(f"Failed to create ResNet50: {e}")
            model = PlaceholderModel(num_classes)
            
    elif model_name == 'placeholder':
        # Use placeholder model for testing
        model = PlaceholderModel(num_classes)
        logger.info(f"Created PlaceholderModel with {num_classes} classes")
        
    else:
        logger.warning(f"Unknown model name '{model_name}', using PlaceholderModel")
        model = PlaceholderModel(num_classes)
    
    return model
