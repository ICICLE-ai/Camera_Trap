"""
Metrics calculation utilities
"""

import numpy as np
from typing import Dict, List, Any
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.n_classes = len(class_names)
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray,
                          losses: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics.

        Args:
            predictions: Predicted class indices
            labels: True class indices
            losses: Per-sample losses (optional)

        Returns:
            Dictionary of calculated metrics
        """

        metrics: Dict[str, Any] = {}

        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        # Custom balanced accuracy without sklearn warning when some classes are missing in labels
        metrics['balanced_accuracy'] = float(self._balanced_accuracy(labels, predictions))

        # Per-class metrics with zero_division handling
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(self.n_classes)),  # ensure consistent class set
            average=None,
            zero_division=0,
        )

        # Also get macro and weighted averages with zero_division handling
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(self.n_classes)),
            average='macro',
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(self.n_classes)),
            average='weighted',
            zero_division=0,
        )

        metrics['per_class'] = {}
        for i in range(self.n_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            # Support arrays are length n_classes thanks to labels=...
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0,
            }

        # Macro and weighted averages (using computed values)
        metrics['macro_precision'] = float(precision_macro)
        metrics['macro_recall'] = float(recall_macro)
        metrics['macro_f1'] = float(f1_macro)

        # Weighted averages (using computed values)
        metrics['weighted_precision'] = float(precision_weighted)
        metrics['weighted_recall'] = float(recall_weighted)
        metrics['weighted_f1'] = float(f1_weighted)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=list(range(self.n_classes)))
        metrics['confusion_matrix'] = cm.tolist()

        # Loss metrics if provided
        if losses is not None:
            metrics['mean_loss'] = float(np.mean(losses))
            metrics['std_loss'] = float(np.std(losses))

        # Additional metrics
        metrics['total_samples'] = int(len(labels))
        metrics['class_distribution'] = self._get_class_distribution(labels)

        return metrics

    def _balanced_accuracy(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute balanced accuracy across known classes without raising sklearn warnings.
        - Uses confusion_matrix with fixed label set [0..n_classes-1].
        - Averages recall only over classes present in labels (support > 0).
        """
        if labels.size == 0:
            return 0.0
        cm = confusion_matrix(labels, predictions, labels=list(range(self.n_classes)))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_support = cm.sum(axis=1)
            true_positives = np.diag(cm)
            recall = np.divide(true_positives, per_class_support, out=np.zeros_like(true_positives, dtype=float), where=per_class_support > 0)
        present_mask = per_class_support > 0
        if not np.any(present_mask):
            return 0.0
        return float(np.mean(recall[present_mask]))
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Get distribution of samples across classes."""
        unique, counts = np.unique(labels, return_counts=True)
        
        distribution = {}
        for class_idx, count in zip(unique, counts):
            if class_idx < len(self.class_names):
                distribution[self.class_names[class_idx]] = int(count)
        
        return distribution
    
    def print_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Print metrics in a readable format."""
        
        if prefix:
            prefix = f"{prefix} "
            
        logger.info(f"{prefix}Metrics Summary:")
        logger.info(f"  Total samples: {metrics['total_samples']}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1: {metrics.get('weighted_f1', 0.0):.4f}")
        
        if 'mean_loss' in metrics:
            logger.info(f"  Mean Loss: {metrics['mean_loss']:.4f}")
        
        # Per-class performance
        logger.info("  Per-class Performance:")
        for class_name, class_metrics in metrics.get('per_class', {}).items():
            logger.info(f"    {class_name}: "
                       f"P={class_metrics['precision']:.3f}, "
                       f"R={class_metrics['recall']:.3f}, "
                       f"F1={class_metrics['f1']:.3f}, "
                       f"N={class_metrics['support']}")
    
    def compare_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                       name1: str = "Model 1", name2: str = "Model 2"):
        """Compare metrics between two models."""
        
        logger.info(f"Comparison: {name1} vs {name2}")
        
        # Key metrics to compare
        key_metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
        
        for metric in key_metrics:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric] 
                diff = val2 - val1
                improvement = "↑" if diff > 0 else "↓" if diff < 0 else "="
                
                logger.info(f"  {metric}: {val1:.4f} → {val2:.4f} ({diff:+.4f}) {improvement}")
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple evaluations."""
        
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Simple metrics that can be averaged
        avg_metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mean_loss']
        
        for metric in avg_metrics:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                aggregated[f'mean_{metric}'] = float(np.mean(values))
                aggregated[f'std_{metric}'] = float(np.std(values))
                aggregated[f'min_{metric}'] = float(np.min(values))
                aggregated[f'max_{metric}'] = float(np.max(values))
        
        # Total samples
        aggregated['total_evaluations'] = len(metrics_list)
        aggregated['total_samples'] = sum(m.get('total_samples', 0) for m in metrics_list)
        
        return aggregated
