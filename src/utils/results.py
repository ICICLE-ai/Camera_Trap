"""
Results management and storage utilities
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages experiment results storage and retrieval."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.json"
        self.results = {
            'experiment_info': {},
            'checkpoint_results': {},
            'summary': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
        }
    
    def set_experiment_info(self, camera: str, mode: str, config: Dict[str, Any]):
        """Set experiment information."""
        self.results['experiment_info'] = {
            'camera': camera,
            'mode': mode,
            'config_file': config.get('config', ''),
            'mode_type': self._extract_mode_type(config.get('config', '')),
            'model_version': config.get('model_version', 'v1'),
            'seed': config.get('seed', 42),
            'device': config.get('device', 'cuda'),
            'timestamp': datetime.now().isoformat()
        }
    
    def add_checkpoint_result(self, checkpoint: str, metrics: Dict[str, float], 
                            predictions: Optional[List] = None, 
                            labels: Optional[List] = None,
                            sample_count: int = 0):
        """Add results for a single checkpoint."""
        checkpoint_data = {
            'metrics': {
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'balanced_accuracy': float(metrics.get('balanced_accuracy', 0.0)),
                'loss': float(metrics.get('loss', 0.0)),
                'sample_count': sample_count
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store predictions and labels if provided
        if predictions is not None:
            self._save_predictions(checkpoint, predictions, labels)
            checkpoint_data['predictions_file'] = f"predictions_{checkpoint}.json"
        
        self.results['checkpoint_results'][checkpoint] = checkpoint_data
        logger.debug(f"Added results for checkpoint {checkpoint}")
    
    def calculate_summary(self):
        """Calculate summary statistics across all checkpoints."""
        checkpoint_results = self.results['checkpoint_results']
        
        if not checkpoint_results:
            logger.warning("No checkpoint results to summarize")
            return
        
        # Extract metrics from all checkpoints
        accuracies = []
        balanced_accuracies = []
        losses = []
        sample_counts = []
        
        for checkpoint_data in checkpoint_results.values():
            metrics = checkpoint_data['metrics']
            accuracies.append(metrics['accuracy'])
            balanced_accuracies.append(metrics['balanced_accuracy'])
            if metrics['loss'] > 0:  # Only include non-zero losses
                losses.append(metrics['loss'])
            sample_counts.append(metrics['sample_count'])
        
        # Calculate summary statistics
        summary = {
            'num_checkpoints': len(checkpoint_results),
            'total_samples': sum(sample_counts),
            'average_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'average_balanced_accuracy': float(np.mean(balanced_accuracies)),
            'std_balanced_accuracy': float(np.std(balanced_accuracies)),
            'min_balanced_accuracy': float(np.min(balanced_accuracies)),
            'max_balanced_accuracy': float(np.max(balanced_accuracies))
        }
        
        # Add loss statistics if available
        if losses:
            summary.update({
                'average_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses)),
                'min_loss': float(np.min(losses)),
                'max_loss': float(np.max(losses))
            })
        
        # Performance assessment
        avg_bal_acc = summary['average_balanced_accuracy']
        if avg_bal_acc >= 0.8:
            performance_level = "Excellent"
        elif avg_bal_acc >= 0.6:
            performance_level = "Good"
        elif avg_bal_acc >= 0.4:
            performance_level = "Fair"
        else:
            performance_level = "Needs Improvement"
        
        summary['performance_level'] = performance_level
        
        self.results['summary'] = summary
        logger.info(f"Calculated summary statistics for {len(checkpoint_results)} checkpoints")
    
    def save_results(self) -> str:
        """Save results to JSON file."""
        try:
            # Update metadata
            self.results['metadata']['saved_at'] = datetime.now().isoformat()
            
            # Save with pretty printing
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2, sort_keys=False)
            
            logger.info(f"Results saved to {self.results_file}")
            return str(self.results_file)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def load_results(self, results_file: Optional[str] = None) -> Dict[str, Any]:
        """Load results from JSON file."""
        file_path = Path(results_file) if results_file else self.results_file
        
        try:
            with open(file_path, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Results loaded from {file_path}")
            return self.results
        except Exception as e:
            logger.error(f"Failed to load results from {file_path}: {e}")
            raise
    
    def get_best_checkpoint(self, metric: str = 'balanced_accuracy') -> Optional[str]:
        """Get the best performing checkpoint based on specified metric."""
        checkpoint_results = self.results['checkpoint_results']
        
        if not checkpoint_results:
            return None
        
        best_checkpoint = None
        best_score = -1
        
        for checkpoint, data in checkpoint_results.items():
            score = data['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def get_worst_checkpoint(self, metric: str = 'balanced_accuracy') -> Optional[str]:
        """Get the worst performing checkpoint based on specified metric."""
        checkpoint_results = self.results['checkpoint_results']
        
        if not checkpoint_results:
            return None
        
        worst_checkpoint = None
        worst_score = float('inf')
        
        for checkpoint, data in checkpoint_results.items():
            score = data['metrics'].get(metric, float('inf'))
            if score < worst_score:
                worst_score = score
                worst_checkpoint = checkpoint
        
        return worst_checkpoint
    
    def get_checkpoint_metrics(self, checkpoint: str) -> Optional[Dict[str, float]]:
        """Get metrics for a specific checkpoint."""
        checkpoint_data = self.results['checkpoint_results'].get(checkpoint)
        if checkpoint_data:
            return checkpoint_data['metrics']
        return None
    
    def export_summary_csv(self, filename: str = "summary.csv"):
        """Export summary results to CSV file."""
        import pandas as pd
        
        checkpoint_results = self.results['checkpoint_results']
        if not checkpoint_results:
            logger.warning("No results to export")
            return
        
        # Prepare data for CSV
        rows = []
        for checkpoint, data in checkpoint_results.items():
            row = {'checkpoint': checkpoint}
            row.update(data['metrics'])
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Results exported to CSV: {csv_path}")
    
    def _save_predictions(self, checkpoint: str, predictions: List, labels: Optional[List] = None):
        """Save predictions for a checkpoint."""
        pred_file = self.output_dir / f"predictions_{checkpoint}.json"
        
        pred_data = {
            'checkpoint': checkpoint,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        if labels is not None:
            pred_data['labels'] = labels
        
        try:
            with open(pred_file, 'w') as f:
                json.dump(pred_data, f, indent=2, default=self._json_serializer)
            logger.debug(f"Predictions saved for checkpoint {checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to save predictions for {checkpoint}: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    def _extract_mode_type(self, config_path: str) -> str:
        """Extract mode type from config path."""
        if not config_path:
            return 'unknown'
        
        if 'zs.yaml' in config_path:
            return 'zs'
        elif 'oracle.yaml' in config_path:
            return 'oracle'
        elif 'accumulative.yaml' in config_path:
            return 'accumulative'
        else:
            return Path(config_path).stem


def aggregate_results(results_files: List[str]) -> Dict[str, Any]:
    """
    Aggregate results from multiple experiment files.
    
    Args:
        results_files: List of results file paths
        
    Returns:
        Aggregated results dictionary
    """
    aggregated = {
        'experiments': [],
        'overall_summary': {},
        'metadata': {
            'aggregated_at': datetime.now().isoformat(),
            'num_experiments': len(results_files)
        }
    }
    
    all_accuracies = []
    all_balanced_accuracies = []
    
    for results_file in results_files:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            aggregated['experiments'].append({
                'file': results_file,
                'experiment_info': results.get('experiment_info', {}),
                'summary': results.get('summary', {})
            })
            
            # Collect metrics for overall summary
            summary = results.get('summary', {})
            if 'average_accuracy' in summary:
                all_accuracies.append(summary['average_accuracy'])
            if 'average_balanced_accuracy' in summary:
                all_balanced_accuracies.append(summary['average_balanced_accuracy'])
                
        except Exception as e:
            logger.warning(f"Failed to load results from {results_file}: {e}")
    
    # Calculate overall statistics
    if all_accuracies:
        aggregated['overall_summary']['mean_accuracy'] = float(np.mean(all_accuracies))
        aggregated['overall_summary']['std_accuracy'] = float(np.std(all_accuracies))
    
    if all_balanced_accuracies:
        aggregated['overall_summary']['mean_balanced_accuracy'] = float(np.mean(all_balanced_accuracies))
        aggregated['overall_summary']['std_balanced_accuracy'] = float(np.std(all_balanced_accuracies))
    
    logger.info(f"Aggregated results from {len(results_files)} experiments")
    return aggregated
