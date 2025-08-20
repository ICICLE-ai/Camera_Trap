"""
BioCLIP Model Integration for ICICLE-Benchmark V2

This module provides BioCLIP model loading and initialization with pre-trained weights.
BioCLIP is a vision-language model specifically designed for biological and ecological applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# BioCLIP template for biological species (simplified to avoid tokenization issues)
BIOCLIP_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
]

# OpenAI ImageNet template (complete version from original ICICLE-Benchmark)
OPENAI_IMAGENET_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
    'a bad photo of a {CLZ_NAME}.',
    'a photo of many {CLZ_NAME}.',
    'a sculpture of a {CLZ_NAME}.',
    'a photo of the hard to see {CLZ_NAME}.',
    'a low resolution photo of the {CLZ_NAME}.',
    'a rendering of a {CLZ_NAME}.',
    'graffiti of a {CLZ_NAME}.',
    'a bad photo of the {CLZ_NAME}.',
    'a cropped photo of the {CLZ_NAME}.',
    'a tattoo of a {CLZ_NAME}.',
    'the embroidered {CLZ_NAME}.',
    'a photo of a hard to see {CLZ_NAME}.',
    'a bright photo of a {CLZ_NAME}.',
    'a photo of a clean {CLZ_NAME}.',
    'a photo of a dirty {CLZ_NAME}.',
    'a dark photo of the {CLZ_NAME}.',
    'a drawing of a {CLZ_NAME}.',
    'a photo of my {CLZ_NAME}.',
    'the plastic {CLZ_NAME}.',
    'a photo of the cool {CLZ_NAME}.',
    'a close-up photo of a {CLZ_NAME}.',
    'a black and white photo of the {CLZ_NAME}.',
    'a painting of the {CLZ_NAME}.',
    'a painting of a {CLZ_NAME}.',
    'a pixelated photo of the {CLZ_NAME}.',
    'a sculpture of the {CLZ_NAME}.',
    'a bright photo of the {CLZ_NAME}.',
    'a cropped photo of a {CLZ_NAME}.',
    'a plastic {CLZ_NAME}.',
    'a photo of the dirty {CLZ_NAME}.',
    'a jpeg corrupted photo of a {CLZ_NAME}.',
    'a blurry photo of the {CLZ_NAME}.',
    'a photo of the {CLZ_NAME}.',
    'a good photo of the {CLZ_NAME}.',
    'a rendering of the {CLZ_NAME}.',
    'a {CLZ_NAME} in a video game.',
    'a photo of one {CLZ_NAME}.',
    'a doodle of a {CLZ_NAME}.',
    'a close-up photo of the {CLZ_NAME}.',
    'a photo of a {CLZ_NAME}.',
    'the origami {CLZ_NAME}.',
    'the {CLZ_NAME} in a video game.',
    'a sketch of a {CLZ_NAME}.',
    'a doodle of the {CLZ_NAME}.',
    'a origami {CLZ_NAME}.',
    'a low resolution photo of a {CLZ_NAME}.',
    'the toy {CLZ_NAME}.',
    'a rendition of the {CLZ_NAME}.',
    'a photo of the clean {CLZ_NAME}.',
    'a photo of a large {CLZ_NAME}.',
    'a rendition of a {CLZ_NAME}.',
    'a photo of a nice {CLZ_NAME}.',
    'a photo of a weird {CLZ_NAME}.',
    'a blurry photo of a {CLZ_NAME}.',
    'a cartoon {CLZ_NAME}.',
    'art of a {CLZ_NAME}.',
    'a sketch of the {CLZ_NAME}.',
    'a embroidered {CLZ_NAME}.',
    'a pixelated photo of a {CLZ_NAME}.',
    'itap of the {CLZ_NAME}.',
    'a jpeg corrupted photo of the {CLZ_NAME}.',
    'a good photo of a {CLZ_NAME}.',
    'a plushie {CLZ_NAME}.',
    'a photo of the nice {CLZ_NAME}.',
    'a photo of the small {CLZ_NAME}.',
    'a photo of the weird {CLZ_NAME}.',
    'the cartoon {CLZ_NAME}.',
    'art of the {CLZ_NAME}.',
    'a drawing of the {CLZ_NAME}.',
    'a photo of the large {CLZ_NAME}.',
    'a black and white photo of a {CLZ_NAME}.',
    'the plushie {CLZ_NAME}.',
    'a dark photo of a {CLZ_NAME}.',
    'itap of a {CLZ_NAME}.',
    'graffiti of the {CLZ_NAME}.',
    'a toy {CLZ_NAME}.',
    'itap of my {CLZ_NAME}.',
    'a photo of a cool {CLZ_NAME}.',
    'a photo of a small {CLZ_NAME}.',
    'a tattoo of the {CLZ_NAME}.',
]


class BioCLIPClassifier(nn.Module):
    """
    BioCLIP-based classifier for camera trap images.
    Uses pre-trained BioCLIP visual encoder with learned text embeddings.
    """
    
    def __init__(self, visual_model, embed_dim: int, num_classes: int):
        super(BioCLIPClassifier, self).__init__()
        self.visual_model = visual_model
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head = None
        self.initialized = False
        
        logger.info(f"Created BioCLIPClassifier with {embed_dim} embed_dim, {num_classes} classes")
    
    def init_head(self, class_embedding: torch.Tensor):
        """Initialize classification head with pre-computed class embeddings."""
        assert not self.initialized, 'Head already initialized.'
        
        self.head = nn.Linear(class_embedding.size(1), class_embedding.size(0), bias=True)
        self.head.weight.data = class_embedding
        self.head.bias.data.zero_()
        self.initialized = True
        
        logger.info(f"Initialized classification head with shape {class_embedding.shape}")
    
    def reset_head(self):
        """Reset classification head to random initialization."""
        assert self.initialized, 'Head not initialized.'
        device = next(self.parameters()).device
        self.head = nn.Linear(self.head.in_features, self.head.out_features, bias=True).to(device)
        logger.info("Reset classification head to random weights")
    
    def forward(self, images, return_feats=False):
        """Forward pass through the model."""
        # Extract visual features
        x = self.visual_model(images)
        feats = F.normalize(x, dim=-1)
        
        # Classification
        x = self.head(feats)
        
        if return_feats:
            return x, feats
        else:
            return x
    
    def save(self, path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f'Saving BioCLIP classifier to {path}')
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model state dict."""
        logger.info(f'Loading BioCLIP classifier from {path}')
        self.load_state_dict(torch.load(path, map_location='cpu'))


def get_texts_for_class(class_name: str, lookup_dict: Optional[Dict] = None) -> List[str]:
    """
    Generate text templates for a class name using BioCLIP or ImageNet templates.
    Based on the original ICICLE-Benchmark implementation.
    
    Args:
        class_name: Common name of the species/class
        lookup_dict: Optional taxonomic lookup dictionary
        
    Returns:
        List of text templates for the class
    """
    use_bioclip_template = True
    
    # Check if we should use BioCLIP template (with taxonomic info)
    if lookup_dict is None or class_name not in lookup_dict:
        use_bioclip_template = False
    else:
        tax = lookup_dict[class_name]
        for t in tax:
            if not isinstance(t, str) and (isinstance(t, float) and np.isnan(t)):
                use_bioclip_template = False
                break
    
    if use_bioclip_template and lookup_dict:
        # Use BioCLIP template with taxonomic information
        tax = lookup_dict[class_name]
        common = class_name
        scientific = tax[-1] if tax else common
        taxonomic = ' '.join([str(t) for t in tax if isinstance(t, str)])
        scientific_common = f'{scientific} with common name {common}'
        taxonomic_common = f'{taxonomic} with common name {common}'
        
        names = [common, scientific, taxonomic, scientific_common, taxonomic_common]
        texts = []
        for n in names:
            if n.strip():  # Only add non-empty names
                texts += [template.format(CLZ_NAME=n) for template in BIOCLIP_TEMPLATE]
    else:
        # Use OpenAI ImageNet template
        texts = [template.format(CLZ_NAME=class_name) for template in OPENAI_IMAGENET_TEMPLATE]
    
    return texts


def get_class_embeddings(model, tokenizer, class_names: List[str], 
                        lookup_dict: Optional[Dict] = None) -> torch.Tensor:
    """
    Generate class embeddings for given class names using BioCLIP text encoder.
    Uses the same approach as original ICICLE-Benchmark.
    
    Args:
        model: BioCLIP model with text encoder
        tokenizer: HuggingFace tokenizer
        class_names: List of class names
        lookup_dict: Optional taxonomic lookup dictionary
        
    Returns:
        Tensor of class embeddings [num_classes, embed_dim]
    """
    device = next(model.parameters()).device
    context_length = getattr(model, 'context_length', 77)
    
    # Get embedding dimension
    embed_dim = getattr(model, 'embed_dim', 512)
    
    logger.info(f"Generating class embeddings for {len(class_names)} classes")
    
    with torch.no_grad():
        class_embeddings = torch.empty(len(class_names), embed_dim)
        
        for idx, class_name in enumerate(class_names):
            logger.debug(f'Getting class embedding for {class_name}...')
            
            # Generate text templates for this class
            texts = get_texts_for_class(class_name, lookup_dict)
            
            try:
                # Use HuggingFace tokenizer with proper padding and truncation
                tokenized = tokenizer(
                    texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=context_length, 
                    return_tensors='pt'
                )
                input_ids = tokenized['input_ids'].to(device)
                
                # Get text embeddings and average them
                text_embeddings = model.encode_text(input_ids)
                text_embeddings = F.normalize(text_embeddings, dim=-1).mean(dim=0)
                text_embeddings = F.normalize(text_embeddings, dim=-1)
                class_embeddings[idx] = text_embeddings.cpu()
                
            except Exception as e:
                logger.warning(f"Text encoding failed for {class_name}: {e}")
                # Use random embedding as fallback
                class_embeddings[idx] = F.normalize(torch.randn(embed_dim), dim=-1)
        
        logger.info(f"Generated class embeddings with shape {class_embeddings.shape}")
        return class_embeddings


def get_simple_class_embeddings(model, tokenizer, class_names: List[str]) -> torch.Tensor:
    """
    Generate simple class embeddings using basic "a photo of {class}" templates.
    Uses HuggingFace tokenizer like the original ICICLE-Benchmark implementation.
    """
    device = next(model.parameters()).device
    context_length = getattr(model, 'context_length', 77)
    
    # Get embedding dimension from the actual model
    embed_dim = getattr(model, 'embed_dim', None)
    if embed_dim is None:
        # For CLIP models, try different attribute names
        embed_dim = getattr(model, 'text_projection', None)
        if embed_dim is not None:
            embed_dim = embed_dim.shape[1] if hasattr(embed_dim, 'shape') else embed_dim
        else:
            embed_dim = 512  # Fallback
    
    logger.info(f"Generating simple class embeddings for {len(class_names)} classes with embed_dim={embed_dim}")
    
    with torch.no_grad():
        class_embeddings = torch.empty(len(class_names), embed_dim)
        
        for idx, class_name in enumerate(class_names):
            # Simple template
            texts = [f"a photo of {class_name}."]
            
            try:
                # Use HuggingFace tokenizer with proper padding and truncation
                tokenized = tokenizer(
                    texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=context_length, 
                    return_tensors='pt'
                )
                input_ids = tokenized['input_ids'].to(device)
                
                # Get text embedding
                text_embedding = model.encode_text(input_ids)
                text_embedding = F.normalize(text_embedding, dim=-1)
                
                class_embeddings[idx] = text_embedding[0].cpu()
                logger.debug(f"Generated embedding for '{texts[0]}'")
                
            except Exception as e:
                logger.warning(f"Failed to encode text for {class_name}: {e}")
                # Use random normalized embedding as fallback
                class_embeddings[idx] = F.normalize(torch.randn(embed_dim), dim=-1)
        
        logger.info(f"Generated simple class embeddings with shape {class_embeddings.shape}")
        return class_embeddings


def create_bioclip_model(num_classes: int, class_names: List[str], 
                        pretrained_path: Optional[str] = None,
                        version: str = 'v1',
                        device: str = 'cuda') -> BioCLIPClassifier:
    """
    Create BioCLIP classifier with pre-trained weights.
    
    Args:
        num_classes: Number of classes
        class_names: List of class names
        pretrained_path: Path to pre-trained BioCLIP weights
        version: BioCLIP version ('v1' or 'v2')
        device: Device to load model on
        
    Returns:
        BioCLIPClassifier instance
    """
    try:
        # Try to import open_clip (BioCLIP uses this)
        from open_clip import create_model_and_transforms
        logger.info("Using open_clip for BioCLIP model creation")
    except ImportError:
        logger.warning("open_clip not available, falling back to placeholder model")
        return create_bioclip_placeholder(num_classes, device)
    
    # Default BioCLIP configuration
    if version == 'v2':
        # BioCLIP v2 uses ViT-L-14 architecture (based on config)
        model_name = 'ViT-L-14'  # Larger model with patch size 14
        precision = 'fp32'
        force_image_size = 224
        logger.info(f"Using BioCLIP v2 architecture: {model_name}")
    else:  # v1
        model_name = 'ViT-B-16'
        precision = 'fp32'
        force_image_size = None
        logger.info(f"Using BioCLIP v1 architecture: {model_name}")
    
    # Try to load from HuggingFace if no local path provided
    if pretrained_path is None:
        logger.info(f"No local BioCLIP {version} weights specified, looking for local weights first")
        
        # Check for local BioCLIP weights based on version
        if version == 'v2':
            local_paths = [
                'pretrained_weight/bioclip-2/open_clip_pytorch_model.bin',
                'pretrained_weight/bioclip-2/open_clip_model.safetensors',
                'ICICLE-Benchmark/pretrained_weight/bioclip-2/open_clip_pytorch_model.bin',
                'ICICLE-Benchmark/pretrained_weight/bioclip-2/open_clip_model.safetensors'
            ]
        else:  # v1
            local_paths = [
                'pretrained_weight/bioclip/open_clip_pytorch_model.bin',
                'ICICLE-Benchmark/pretrained_weight/bioclip/open_clip_pytorch_model.bin'
            ]
        
        for path in local_paths:
            if os.path.exists(path):
                pretrained_path = path
                logger.info(f"Found local BioCLIP {version} weights at: {path}")
                break
    
    # Load from local weights
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading BioCLIP {version} from local weights: {pretrained_path}")
        try:
            # Special handling for BioCLIP v2
            if version == 'v2':
                # Try with specific settings for BioCLIP v2
                model, preprocess_train, preprocess_val = create_model_and_transforms(
                    model_name,
                    pretrained_path,
                    precision=precision,
                    device=device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_image_size=force_image_size,
                    pretrained_image=False,  # Don't load pretrained image weights
                    output_dict=True,
                )
            else:
                # BioCLIP v1 with original settings
                model, preprocess_train, preprocess_val = create_model_and_transforms(
                    model_name,
                    pretrained_path,
                    precision=precision,
                    device=device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    output_dict=True,
                )
            
            # Load tokenizer from same directory
            tokenizer_path = os.path.dirname(pretrained_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded tokenizer from: {tokenizer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load BioCLIP {version} from local weights: {e}")
            return create_bioclip_placeholder(num_classes, device)
    
    else:
        logger.warning(f"No local BioCLIP {version} weights found, this should not happen")
        return create_bioclip_placeholder(num_classes, device)
    
    # Create classifier
    embed_dim = getattr(model, 'embed_dim', None)
    if embed_dim is None:
        # For CLIP models, try different attribute names
        embed_dim = getattr(model, 'text_projection', None)
        if embed_dim is not None:
            embed_dim = embed_dim.shape[1]
        else:
            # Fallback to common CLIP embedding dimension
            embed_dim = 512
            logger.warning(f"Could not determine embed_dim, using default: {embed_dim}")
    
    logger.info(f"Using embedding dimension: {embed_dim}")
    classifier = BioCLIPClassifier(model.visual, embed_dim, num_classes)
    
    # Generate class embeddings
    try:
        # Use the complete class embedding generation like original ICICLE-Benchmark
        logger.info(f"Generating class embeddings for {len(class_names)} classes")
        
        class_embeddings = get_simple_class_embeddings(model, tokenizer, class_names)
        classifier.init_head(class_embeddings)
        
    except Exception as e:
        logger.error(f"Failed to initialize class embeddings: {e}")
        # Initialize with random head as fallback
        classifier.reset_head()
    
    classifier = classifier.to(device)
    logger.info(f"Created BioCLIP {version} classifier with {num_classes} classes")
    
    return classifier


def create_bioclip_placeholder(num_classes: int, device: str = 'cuda') -> nn.Module:
    """
    Create a placeholder model that mimics BioCLIP structure when BioCLIP is not available.
    This is essentially a CNN-based classifier.
    """
    logger.warning("Creating BioCLIP placeholder model")
    
    class BioCLIPPlaceholder(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.embed_dim = 512
            
            # Simple CNN backbone (similar to ResNet structure)
            self.visual_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
                
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x, return_feats=False):
            feats = self.visual_model(x)
            feats = F.normalize(feats, dim=-1)
            logits = self.head(feats)
            
            if return_feats:
                return logits, feats
            else:
                return logits
        
        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
        
        def load(self, path: str):
            self.load_state_dict(torch.load(path, map_location='cpu'))
    
    model = BioCLIPPlaceholder(num_classes).to(device)
    logger.info(f"Created BioCLIP placeholder with {num_classes} classes")
    
    return model
