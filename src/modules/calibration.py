"""
Calibration Module for inference-level model calibration
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import torch
import torch.nn as nn

# Avoid runtime dependency on removed core.config; use type-only alias
if TYPE_CHECKING:
    Config = Any

logger = logging.getLogger(__name__)


class CalibrationFactory:
    """Factory for creating calibration modules."""
    
    @staticmethod
    def create(config: 'Config') -> Optional['BaseCalibrationModule']:
        """Create calibration module if calibration is enabled."""
        
        if not config.calibration:
            return None
        
        method = config.calibration_params.get('method', 'temperature_scaling')
        
        if method == 'temperature_scaling':
            return TemperatureScaling(config)
        elif method == 'platt_scaling':
            return PlattScaling(config)
        elif method == 'isotonic':
            return IsotonicCalibration(config)
        else:
            logger.warning(f"Unknown calibration method: {method}, using temperature_scaling")
            return TemperatureScaling(config)


class BaseCalibrationModule:
    """Base class for calibration modules."""
    
    def __init__(self, config: 'Config'):
        self.config = config
        self.device = config.device
        self.is_fitted = False
    
    def process(self, model: nn.Module, eval_dataset) -> Dict[str, Any]:
        """
        Calibrate model predictions and return calibrated results.
        
        Args:
            model: Trained model
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary with calibrated predictions and metrics
        """
        
        # Get raw predictions
        raw_logits, true_labels = self._get_predictions(model, eval_dataset)
        
        # Fit calibration if not already done
        if not self.is_fitted:
            self._fit_calibration(raw_logits, true_labels)
            self.is_fitted = True
        
        # Apply calibration
        calibrated_probs = self._apply_calibration(raw_logits)
        calibrated_preds = np.argmax(calibrated_probs, axis=1)
        
        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            calibrated_probs, true_labels
        )
        
        return {
            'calibrated_predictions': calibrated_preds,
            'calibrated_probabilities': calibrated_probs,
            'raw_logits': raw_logits,
            'true_labels': true_labels,
            'calibration_metrics': calibration_metrics
        }
    
    def _get_predictions(self, model: nn.Module, dataset) -> tuple:
        """Get raw model predictions and true labels."""
        
        model.eval()
        
        # Placeholder implementation
        # In actual implementation, this would:
        # 1. Create dataloader
        # 2. Run inference to get logits
        # 3. Collect true labels
        # 4. Return as numpy arrays
        
        n_samples = len(dataset) if dataset else 100
        n_classes = len(self.config.class_names)
        
        # Simulate logits and labels
        raw_logits = np.random.randn(n_samples, n_classes)
        true_labels = np.random.randint(0, n_classes, n_samples)
        
        return raw_logits, true_labels
    
    def _fit_calibration(self, logits: np.ndarray, labels: np.ndarray):
        """Fit calibration parameters (to be implemented by subclasses)."""
        pass
    
    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to logits (to be implemented by subclasses)."""
        # Default: return softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _calculate_calibration_metrics(self, probs: np.ndarray, 
                                     labels: np.ndarray) -> Dict[str, float]:
        """Calculate calibration-specific metrics."""
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(probs, labels)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(probs, labels)
        
        # Brier Score
        brier_score = self._calculate_brier_score(probs, labels)
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'brier_score': brier_score
        }
    
    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        
        max_probs = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _calculate_mce(self, probs: np.ndarray, labels: np.ndarray,
                      n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        
        max_probs = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return float(mce)
    
    def _calculate_brier_score(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier Score."""
        
        n_classes = probs.shape[1]
        one_hot_labels = np.eye(n_classes)[labels]
        
        brier_score = np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))
        return float(brier_score)


class TemperatureScaling(BaseCalibrationModule):
    """Temperature scaling calibration."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.temperature = 1.0
    
    def _fit_calibration(self, logits: np.ndarray, labels: np.ndarray):
        """Fit temperature parameter using validation data."""
        
        logger.info("Fitting temperature scaling...")
        
        # Convert to tensors for optimization
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.LongTensor(labels)
        
        # Initialize temperature parameter
        temperature = torch.ones(1, requires_grad=True)
        
        # Optimize temperature using LBFGS
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        logger.info(f"Fitted temperature: {self.temperature:.4f}")
    
    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class PlattScaling(BaseCalibrationModule):
    """Platt scaling (sigmoid) calibration."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.A = 1.0
        self.B = 0.0
    
    def _fit_calibration(self, logits: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling parameters."""
        
        logger.info("Fitting Platt scaling...")
        
        # For multi-class, we'll use the max logit as the confidence score
        max_logits = np.max(logits, axis=1)
        predictions = np.argmax(logits, axis=1)
        binary_labels = (predictions == labels).astype(float)
        
        # Fit sigmoid parameters (A, B) such that P(correct) = 1/(1 + exp(A*score + B))
        # This is a simplified implementation
        
        from scipy.optimize import minimize
        
        def sigmoid_loss(params):
            A, B = params
            sigmoid_probs = 1 / (1 + np.exp(A * max_logits + B))
            return -np.mean(binary_labels * np.log(sigmoid_probs + 1e-8) + 
                           (1 - binary_labels) * np.log(1 - sigmoid_probs + 1e-8))
        
        try:
            result = minimize(sigmoid_loss, [1.0, 0.0], method='BFGS')
            self.A, self.B = result.x
            logger.info(f"Fitted Platt parameters: A={self.A:.4f}, B={self.B:.4f}")
        except:
            logger.warning("Failed to fit Platt scaling, using default parameters")
    
    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        
        # This is a simplified version for multi-class
        # In practice, you'd need to fit separate parameters for each class
        
        max_logits = np.max(logits, axis=1)
        calibrated_max_probs = 1 / (1 + np.exp(self.A * max_logits + self.B))
        
        # Convert back to probability distribution
        softmax_probs = self._apply_calibration.__base__(self, logits)  # Get base softmax
        
        # Scale to match calibrated confidence
        scaling_factors = calibrated_max_probs / np.max(softmax_probs, axis=1)
        calibrated_probs = softmax_probs * scaling_factors[:, np.newaxis]
        
        # Renormalize
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
        
        return calibrated_probs


class IsotonicCalibration(BaseCalibrationModule):
    """Isotonic regression calibration."""
    
    def __init__(self, config: 'Config'):
        super().__init__(config)
        self.calibrator = None
    
    def _fit_calibration(self, logits: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression calibration."""
        
        logger.info("Fitting isotonic calibration...")
        
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Use max logit as confidence score
            max_logits = np.max(logits, axis=1)
            predictions = np.argmax(logits, axis=1)
            binary_labels = (predictions == labels).astype(float)
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(max_logits, binary_labels)
            
            logger.info("Isotonic calibration fitted successfully")
            
        except ImportError:
            logger.error("sklearn not available, cannot use isotonic calibration")
            self.calibrator = None
    
    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        
        if self.calibrator is None:
            return super()._apply_calibration(logits)
        
        # Get base softmax probabilities
        base_probs = super()._apply_calibration(logits)
        
        # Apply isotonic calibration to max probabilities
        max_probs = np.max(base_probs, axis=1)
        calibrated_max_probs = self.calibrator.predict(max_probs)
        
        # Scale probabilities to match calibrated confidence
        scaling_factors = calibrated_max_probs / max_probs
        calibrated_probs = base_probs * scaling_factors[:, np.newaxis]
        
        # Renormalize
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
        
        return calibrated_probs
