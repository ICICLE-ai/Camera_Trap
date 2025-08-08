"""
Main Pipeline for ICICLE-Benchmark V2
"""

import logging
import time
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import numpy as np

from ..core.config import Config
from ..data.dataset import DatasetManager
from ..models.factory import ModelFactory
from ..modules.ood import OODFactory
from ..modules.active_learning import ALFactory
from ..modules.continual_learning import CLFactory
from ..modules.calibration import CalibrationFactory
from ..utils.gpu import GPUManager
from ..utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrating the entire ICICLE benchmark workflow."""
    
    def __init__(self, config: Config, gpu_manager: GPUManager = None):
        self.config = config
        self.gpu_manager = gpu_manager or GPUManager(enable_cleanup=False)
        
        # Initialize components
        self.dataset_manager = DatasetManager(config)
        self.model_factory = ModelFactory(config)
        self.metrics_calc = MetricsCalculator(config.class_names)
        
        # Initialize modules
        self.ood_module = OODFactory.create(config.ood_method, config)
        self.al_module = ALFactory.create(config.al_method, config)
        self.cl_module = CLFactory.create(config.cl_method, config)
        self.calibration_module = CalibrationFactory.create(config) if config.calibration else None
        
        # Results storage
        self.results = {}
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting ICICLE-Benchmark V2 Pipeline")
        start_time = time.time()
        
        try:
            # Load model
            model = self._initialize_model()
            
            # Load datasets
            train_datasets, eval_datasets = self._load_datasets()
            
            # Get checkpoint list
            checkpoint_list = self.dataset_manager.get_checkpoint_list()
            logger.info(f"Processing {len(checkpoint_list)} checkpoints: {checkpoint_list}")
            
            # Run pretraining if enabled
            if self.config.pretrain:
                model = self._run_pretraining(model)
            
            # Main checkpoint loop
            for i, checkpoint in enumerate(checkpoint_list):
                logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_list)}: {checkpoint}")
                
                with self.gpu_manager.context_cleanup():
                    checkpoint_results = self._process_checkpoint(
                        model, train_datasets, eval_datasets, checkpoint, i
                    )
                    self.results[checkpoint] = checkpoint_results
                
                # Save intermediate results
                self._save_intermediate_results(checkpoint)
                
                logger.info(f"Completed checkpoint {checkpoint}")
            
            # Final processing
            final_results = self._finalize_results()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _initialize_model(self):
        """Initialize the model."""
        logger.info(f"Initializing model: {self.config.model}")
        model = self.model_factory.create_model()
        
        if torch.cuda.is_available() and self.config.device == 'cuda':
            model = model.to(self.config.device)
            logger.info(f"Model moved to {self.config.device}")
        
        return model
    
    def _load_datasets(self) -> Tuple[Dict, Dict]:
        """Load train and evaluation datasets."""
        logger.info("Loading datasets...")
        
        train_datasets = self.dataset_manager.load_train_datasets()
        eval_datasets = self.dataset_manager.load_eval_datasets()
        
        logger.info(f"Loaded {len(train_datasets)} training datasets")
        logger.info(f"Loaded {len(eval_datasets)} evaluation datasets")
        
        return train_datasets, eval_datasets
    
    def _run_pretraining(self, model):
        """Run pretraining if enabled."""
        logger.info("Running pretraining...")
        
        # Load pretraining dataset
        pretrain_dataset = self.dataset_manager.load_pretrain_dataset()
        
        # Run pretraining using continual learning module
        pretrained_model = self.cl_module.pretrain(
            model, pretrain_dataset, self.config.pretrain_epochs
        )
        
        logger.info("Pretraining completed")
        return pretrained_model
    
    def _process_checkpoint(self, model, train_datasets, eval_datasets, 
                          checkpoint: str, checkpoint_idx: int) -> Dict[str, Any]:
        """Process a single checkpoint."""
        
        # Get datasets for this checkpoint
        train_data = train_datasets[checkpoint]
        eval_data = eval_datasets[checkpoint]
        
        logger.info(f"Training samples: {len(train_data)}, Eval samples: {len(eval_data)}")
        
        # Initialize training mask (all samples initially)
        train_mask = np.ones(len(train_data), dtype=bool)
        
        # 1. OOD Detection
        logger.info("Running OOD detection...")
        model, ood_mask = self.ood_module.process(model, train_data, eval_data, train_mask)
        logger.info(f"OOD selected: {ood_mask.sum()}/{len(ood_mask)} samples")
        
        # 2. Active Learning
        logger.info("Running Active Learning...")
        model, al_mask = self.al_module.process(model, train_data, eval_data, ood_mask, checkpoint)
        logger.info(f"AL selected: {al_mask.sum()}/{len(al_mask)} samples")
        
        # 3. Continual Learning
        logger.info("Running Continual Learning...")
        model = self.cl_module.process(model, train_data, eval_data, al_mask, checkpoint)
        
        # 4. Evaluation
        logger.info("Running evaluation...")
        eval_results = self._evaluate_model(model, eval_data)
        
        # 5. Calibration (if enabled)
        if self.calibration_module:
            logger.info("Running calibration...")
            calibrated_results = self.calibration_module.process(model, eval_data)
            eval_results['calibrated'] = calibrated_results
        
        # Collect checkpoint results
        checkpoint_results = {
            'evaluation': eval_results,
            'ood_mask': ood_mask,
            'al_mask': al_mask,
            'train_samples': len(train_data),
            'eval_samples': len(eval_data),
        }
        
        # Save model if requested
        if not self.config.no_save:
            self._save_model(model, checkpoint)
        
        return checkpoint_results
    
    def _evaluate_model(self, model, eval_dataset) -> Dict[str, Any]:
        """Evaluate model on evaluation dataset."""
        model.eval()
        
        eval_loader = self.dataset_manager.create_dataloader(
            eval_dataset, 
            batch_size=self.config.eval_batch_size,
            shuffle=False
        )
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs, labels = batch
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_losses.append(loss.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        losses = np.concatenate(all_losses)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(predictions, labels, losses)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels,
            'losses': losses
        }
    
    def _save_model(self, model, checkpoint: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, f"model_{checkpoint}.pth")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def _save_intermediate_results(self, checkpoint: str):
        """Save intermediate results for a checkpoint."""
        results_path = os.path.join(self.config.output_dir, f"results_{checkpoint}.pkl")
        
        with open(results_path, 'wb') as f:
            pickle.dump(self.results[checkpoint], f)
        
        logger.debug(f"Intermediate results saved to {results_path}")
    
    def _finalize_results(self) -> Dict[str, Any]:
        """Finalize and save all results."""
        
        # Aggregate results across checkpoints
        final_results = {
            'config': self.config,
            'checkpoint_results': self.results,
            'summary_metrics': self._calculate_summary_metrics()
        }
        
        # Save final results
        final_path = os.path.join(self.config.output_dir, "final_results.pkl")
        with open(final_path, 'wb') as f:
            pickle.dump(final_results, f)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "config.yaml")
        from ..core.config import ConfigManager
        ConfigManager().save_config(self.config, config_path)
        
        logger.info(f"Final results saved to {self.config.output_dir}")
        return final_results
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all checkpoints."""
        
        all_accuracies = []
        all_balanced_accs = []
        checkpoint_names = []
        
        for checkpoint, results in self.results.items():
            metrics = results['evaluation']['metrics']
            all_accuracies.append(metrics['accuracy'])
            all_balanced_accs.append(metrics['balanced_accuracy'])
            checkpoint_names.append(checkpoint)
        
        summary = {
            'final_accuracy': all_accuracies[-1] if all_accuracies else 0.0,
            'final_balanced_accuracy': all_balanced_accs[-1] if all_balanced_accs else 0.0,
            'average_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
            'average_balanced_accuracy': np.mean(all_balanced_accs) if all_balanced_accs else 0.0,
            'accuracy_trend': all_accuracies,
            'balanced_accuracy_trend': all_balanced_accs,
            'checkpoint_names': checkpoint_names
        }
        
        return summary
