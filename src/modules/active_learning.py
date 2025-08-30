"""
Active Learning Module
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch
import torch.nn as nn

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    Config = Any

logger = logging.getLogger(__name__)


class ALFactory:
    """Factory for creating Active Learning modules."""
    
    @staticmethod
    def create(method: str, config: 'Config') -> 'BaseALModule':
        """Create AL module based on method name."""
        
        method = method.lower()
        
        if method == 'none':
            return ALNone(config)
        elif method == 'all':
            return ALAll(config)
        elif method == 'random':
            return ALRandom(config)
        elif method == 'uncertainty':
            return ALUncertainty(config)
        elif method == 'active_ft':
            return ALActiveFT(config)
        elif method == 'mls':
            return ALMLS(config)  # Minimum Logit Score
        elif method == 'msp':
            return ALMSP(config)  # Minimum Softmax Probability
        else:
            logger.warning(f"Unknown AL method: {method}, using 'all'")
            return ALAll(config)


class BaseALModule(ABC):
    """Base class for Active Learning modules."""
    
    def __init__(self, config: 'Config'):
        self.config = config
        self.device = config.device
    
    @abstractmethod
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """
        Process datasets for active learning sample selection.
        
        Args:
            model: Current model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            ood_mask: OOD detection mask
            checkpoint: Current checkpoint name
            
        Returns:
            Tuple of (model, al_mask) where al_mask indicates selected samples
        """
        pass


class ALNone(BaseALModule):
    """No active learning - no samples selected."""
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Return empty selection mask."""
        
        logger.info("AL: No active learning - no samples selected")
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        return model, al_mask


class ALAll(BaseALModule):
    """Select all available samples."""
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Select all samples from OOD mask."""
        
        logger.info("AL: All samples selected")
        al_mask = ood_mask.copy()
        return model, al_mask


class ALRandom(BaseALModule):
    """Random sample selection."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.selection_ratio = config.al_params.get('selection_ratio', 0.5)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Randomly select a fraction of OOD samples."""
        
        logger.info(f"AL: Random selection (ratio={self.selection_ratio})")
        
        # Get indices of OOD samples
        ood_indices = np.where(ood_mask)[0]
        
        # Randomly select subset
        n_select = int(len(ood_indices) * self.selection_ratio)
        selected_indices = np.random.choice(ood_indices, size=n_select, replace=False)
        
        # Create selection mask
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        al_mask[selected_indices] = True
        
        return model, al_mask


class ALUncertainty(BaseALModule):
    """Uncertainty-based active learning."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.selection_ratio = config.al_params.get('selection_ratio', 0.5)
        self.uncertainty_method = config.al_params.get('uncertainty_method', 'entropy')
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Select samples with highest uncertainty."""
        
        logger.info(f"AL: Uncertainty-based selection (method={self.uncertainty_method})")
        
        # Compute uncertainties for OOD samples
        uncertainties = self._compute_uncertainties(model, train_dataset, ood_mask)
        
        # Select top uncertain samples
        al_mask = self._select_top_uncertain(uncertainties, ood_mask)
        
        return model, al_mask
    
    def _compute_uncertainties(self, model: nn.Module, dataset, ood_mask: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores for OOD samples."""
        
        # Placeholder implementation
        dataset_size = len(dataset) if dataset else len(ood_mask)
        uncertainties = np.random.random(dataset_size)
        
        # Only consider OOD samples
        uncertainties[~ood_mask] = 0.0
        
        return uncertainties
    
    def _select_top_uncertain(self, uncertainties: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        """Select samples with highest uncertainty."""
        
        # Get number of samples to select
        n_ood = ood_mask.sum()
        n_select = int(n_ood * self.selection_ratio)
        
        # Get top uncertain indices
        top_indices = np.argsort(uncertainties)[-n_select:]
        
        # Create selection mask
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        al_mask[top_indices] = True
        
        return al_mask


class ALActiveFT(BaseALModule):
    """Active learning with feature-based sampling."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.selection_ratio = config.al_params.get('selection_ratio', 0.5)
        self.feature_dim = config.al_params.get('feature_dim', 512)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Active learning using feature-based diversity sampling."""
        
        logger.info("AL: Active feature-based selection")
        
        # Extract features for OOD samples
        features = self._extract_features(model, train_dataset, ood_mask)
        
        # Perform diverse sampling based on features
        al_mask = self._diverse_sampling(features, ood_mask)
        
        return model, al_mask
    
    def _extract_features(self, model: nn.Module, dataset, ood_mask: np.ndarray) -> np.ndarray:
        """Extract features from model for OOD samples."""
        
        # Placeholder implementation
        n_ood = ood_mask.sum()
        features = np.random.randn(n_ood, self.feature_dim)
        
        return features
    
    def _diverse_sampling(self, features: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        """Perform diverse sampling based on features."""
        
        n_features = len(features)
        n_select = int(n_features * self.selection_ratio)
        
        # Simple diverse sampling using random selection
        # In actual implementation, this would use methods like k-means clustering
        selected_indices = np.random.choice(n_features, size=n_select, replace=False)
        
        # Map back to original indices
        ood_indices = np.where(ood_mask)[0]
        selected_original_indices = ood_indices[selected_indices]
        
        # Create selection mask
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        al_mask[selected_original_indices] = True
        
        return al_mask


class ALMLS(BaseALModule):
    """Minimum Logit Score active learning."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.selection_ratio = config.al_params.get('selection_ratio', 0.5)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Select samples with minimum maximum logit scores."""
        
        logger.info("AL: Minimum Logit Score selection")
        
        # Get logit scores
        logit_scores = self._compute_logit_scores(model, train_dataset, ood_mask)
        
        # Select samples with minimum maximum logit scores
        al_mask = self._select_minimum_scores(logit_scores, ood_mask)
        
        return model, al_mask
    
    def _compute_logit_scores(self, model: nn.Module, dataset, ood_mask: np.ndarray) -> np.ndarray:
        """Compute maximum logit scores for samples."""
        
        # Placeholder implementation
        dataset_size = len(dataset) if dataset else len(ood_mask)
        
        # Simulate logit scores (higher values = more confident)
        max_logit_scores = np.random.exponential(2.0, dataset_size)
        max_logit_scores[~ood_mask] = np.inf  # Exclude non-OOD samples
        
        return max_logit_scores
    
    def _select_minimum_scores(self, scores: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        """Select samples with minimum scores."""
        
        n_ood = ood_mask.sum()
        n_select = int(n_ood * self.selection_ratio)
        
        # Get indices with minimum scores
        min_indices = np.argsort(scores)[:n_select]
        
        # Create selection mask
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        al_mask[min_indices] = True
        
        return al_mask


class ALMSP(BaseALModule):
    """Minimum Softmax Probability active learning."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.selection_ratio = config.al_params.get('selection_ratio', 0.5)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                ood_mask: np.ndarray, checkpoint: str) -> Tuple[nn.Module, np.ndarray]:
        """Select samples with minimum maximum softmax probabilities."""
        
        logger.info("AL: Minimum Softmax Probability selection")
        
        # Get softmax probability scores
        prob_scores = self._compute_probability_scores(model, train_dataset, ood_mask)
        
        # Select samples with minimum maximum probabilities
        al_mask = self._select_minimum_scores(prob_scores, ood_mask)
        
        return model, al_mask
    
    def _compute_probability_scores(self, model: nn.Module, dataset, ood_mask: np.ndarray) -> np.ndarray:
        """Compute maximum softmax probability scores for samples."""
        
        # Placeholder implementation
        dataset_size = len(dataset) if dataset else len(ood_mask)
        
        # Simulate probability scores (higher values = more confident)
        max_prob_scores = np.random.beta(5, 2, dataset_size)  # Skewed towards higher values
        max_prob_scores[~ood_mask] = 1.0  # Exclude non-OOD samples
        
        return max_prob_scores
    
    def _select_minimum_scores(self, scores: np.ndarray, ood_mask: np.ndarray) -> np.ndarray:
        """Select samples with minimum scores."""
        
        n_ood = ood_mask.sum()
        n_select = int(n_ood * self.selection_ratio)
        
        # Get indices with minimum scores
        min_indices = np.argsort(scores)[:n_select]
        
        # Create selection mask
        al_mask = np.zeros_like(ood_mask, dtype=bool)
        al_mask[min_indices] = True
        
        return al_mask
