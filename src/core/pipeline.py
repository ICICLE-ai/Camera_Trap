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
        Run the complete pipeline with proper mode handling.
        
        Modes:
        - zs (zero-shot): Use pretrained model, no training, test on each checkpoint
        - oracle: Train on ALL checkpoints at start, then test on each checkpoint  
        - accum (accumulative): Continual learning - start with ckp_0, accumulate training data
        
        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting ICICLE-Benchmark V2 Pipeline")
        logger.info(f"Mode: {self.config.mode}")
        start_time = time.time()
        
        try:
            # Load model
            model = self._initialize_model()
            
            # Load datasets
            train_datasets, eval_datasets = self._load_datasets()
            
            # Get checkpoint list and convert to 0-based indexing
            raw_checkpoint_list = self.dataset_manager.get_checkpoint_list()
            checkpoint_list = self._convert_to_zero_indexed(raw_checkpoint_list)
            logger.info(f"Processing {len(checkpoint_list)} checkpoints (0-indexed): {list(range(len(checkpoint_list)))}")
            logger.info(f"Raw checkpoint mapping: {dict(enumerate(raw_checkpoint_list))}")
            
            # Run pretraining if enabled
            if self.config.pretrain:
                model = self._run_pretraining(model)
            
            # Mode-specific execution
            if self.config.mode == "zs":
                self.results = self._run_zero_shot_mode(model, eval_datasets, checkpoint_list, raw_checkpoint_list)
            elif self.config.mode == "oracle":
                self.results = self._run_oracle_mode(model, train_datasets, eval_datasets, checkpoint_list, raw_checkpoint_list)
            elif self.config.mode == "accum":
                self.results = self._run_accumulative_mode(model, train_datasets, eval_datasets, checkpoint_list, raw_checkpoint_list)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            # Final processing
            final_results = self._finalize_results()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _convert_to_zero_indexed(self, raw_checkpoint_list: List[str]) -> List[str]:
        """Convert raw checkpoint list to 0-indexed mapping."""
        # Sort the checkpoints numerically (ckp_1, ckp_2, ..., ckp_17)
        def extract_number(ckp_str):
            return int(ckp_str.split('_')[1])
        
        sorted_checkpoints = sorted(raw_checkpoint_list, key=extract_number)
        return sorted_checkpoints
    
    def _run_zero_shot_mode(self, model, eval_datasets, checkpoint_list, raw_checkpoint_list) -> Dict[str, Any]:
        """
        Zero-shot mode: Use pretrained model only, no training.
        Test on each checkpoint sequentially.
        """
        logger.info("="*60)
        logger.info("RUNNING ZERO-SHOT MODE")
        logger.info("Training: NONE (pretrained model only)")
        logger.info(f"Testing: Each checkpoint 0-{len(checkpoint_list)-1}")
        logger.info("="*60)
        
        results = {}
        
        for ckp_idx, raw_ckp in enumerate(checkpoint_list):
            test_ckp_idx = ckp_idx
            
            logger.info(f"\n--- CHECKPOINT {ckp_idx} (Zero-Shot) ---")
            logger.info(f"Training on: NONE (pretrained only)")
            logger.info(f"Testing on: ckp_{test_ckp_idx} (raw: {raw_ckp})")
            
            with self.gpu_manager.context_cleanup():
                # No training, just evaluation
                eval_data = eval_datasets[raw_ckp]
                eval_results = self._evaluate_model(model, eval_data)
                
                checkpoint_results = {
                    'mode': 'zero_shot',
                    'train_checkpoints': [],
                    'test_checkpoint': test_ckp_idx, 
                    'raw_test_checkpoint': raw_ckp,
                    'evaluation': eval_results,
                    'train_samples': 0,
                    'eval_samples': len(eval_data) if eval_data else 0,
                }
                
                results[f'ckp_{ckp_idx}'] = checkpoint_results
                
            logger.info(f"Completed checkpoint {ckp_idx} (zero-shot)")
            
        return results
    
    def _run_oracle_mode(self, model, train_datasets, eval_datasets, checkpoint_list, raw_checkpoint_list) -> Dict[str, Any]:
        """
        Oracle mode: Train on ALL data at beginning, then test on each checkpoint.
        """
        logger.info("="*60)
        logger.info("RUNNING ORACLE MODE")
        logger.info(f"Training: ALL checkpoints 0-{len(checkpoint_list)-1} at start")
        logger.info(f"Testing: Each checkpoint 0-{len(checkpoint_list)-1}")
        logger.info("="*60)
        
        # Step 1: Train on ALL data
        logger.info(f"\n--- ORACLE TRAINING PHASE ---")
        logger.info(f"Training on: ALL checkpoints 0-{len(checkpoint_list)-1}")
        logger.info(f"Raw checkpoints: {checkpoint_list}")
        
        # Combine all training data
        all_train_data = self._combine_datasets([train_datasets[raw_ckp] for raw_ckp in checkpoint_list])
        logger.info(f"Total training samples: {len(all_train_data) if all_train_data else 0}")
        
        # Train model on all data
        model = self._train_model_on_data(model, all_train_data, training_phase="oracle_all")
        
        # Step 2: Test on each checkpoint
        results = {}
        
        for ckp_idx, raw_ckp in enumerate(checkpoint_list):
            test_ckp_idx = ckp_idx
            
            logger.info(f"\n--- CHECKPOINT {ckp_idx} (Oracle Test) ---")
            logger.info(f"Training on: ALL 0-{len(checkpoint_list)-1} (already done)")
            logger.info(f"Testing on: ckp_{test_ckp_idx} (raw: {raw_ckp})")
            
            with self.gpu_manager.context_cleanup():
                eval_data = eval_datasets[raw_ckp]
                eval_results = self._evaluate_model(model, eval_data)
                
                checkpoint_results = {
                    'mode': 'oracle',
                    'train_checkpoints': list(range(len(checkpoint_list))),
                    'test_checkpoint': test_ckp_idx,
                    'raw_test_checkpoint': raw_ckp,
                    'evaluation': eval_results,
                    'train_samples': len(all_train_data) if all_train_data else 0,
                    'eval_samples': len(eval_data) if eval_data else 0,
                }
                
                results[f'ckp_{ckp_idx}'] = checkpoint_results
                
            logger.info(f"Completed checkpoint {ckp_idx} (oracle)")
            
        return results
    
    def _run_accumulative_mode(self, model, train_datasets, eval_datasets, checkpoint_list, raw_checkpoint_list) -> Dict[str, Any]:
        """
        Accumulative (Continual Learning) mode:
        - Start: Test on ckp_0 (same as zero-shot)
        - ckp_1: Train on ckp_0, test on ckp_1  
        - ckp_2: Train on ckp_0+1, test on ckp_2
        - ...
        """
        logger.info("="*60)
        logger.info("RUNNING ACCUMULATIVE MODE (Continual Learning)")
        logger.info("Checkpoint 0: Test only (same as zero-shot)")
        logger.info("Checkpoint N: Train on 0..N-1, test on N")
        logger.info("="*60)
        
        results = {}
        accumulated_train_data = []
        
        for ckp_idx, raw_ckp in enumerate(checkpoint_list):
            test_ckp_idx = ckp_idx
            
            if ckp_idx == 0:
                # First checkpoint: zero-shot (no training data available yet)
                logger.info(f"\n--- CHECKPOINT {ckp_idx} (Accumulative - Zero Shot) ---")
                logger.info(f"Training on: NONE (no prior data available)")
                logger.info(f"Testing on: ckp_{test_ckp_idx} (raw: {raw_ckp})")
                
                with self.gpu_manager.context_cleanup():
                    eval_data = eval_datasets[raw_ckp]
                    eval_results = self._evaluate_model(model, eval_data)
                    
                    checkpoint_results = {
                        'mode': 'accumulative',
                        'train_checkpoints': [],
                        'test_checkpoint': test_ckp_idx,
                        'raw_test_checkpoint': raw_ckp,
                        'evaluation': eval_results,
                        'train_samples': 0,
                        'eval_samples': len(eval_data) if eval_data else 0,
                    }
                    
                    results[f'ckp_{ckp_idx}'] = checkpoint_results
                
                # Add this checkpoint's data to accumulated training data for next iteration
                current_train_data = train_datasets[raw_ckp]
                if current_train_data:
                    accumulated_train_data.append(current_train_data)
                    
            else:
                # Subsequent checkpoints: train on accumulated data, test on current
                train_ckp_range = list(range(ckp_idx))  # 0 to ckp_idx-1
                
                logger.info(f"\n--- CHECKPOINT {ckp_idx} (Accumulative - Train & Test) ---")
                logger.info(f"Training on: ckp_{train_ckp_range} (accumulated: 0-{ckp_idx-1})")
                logger.info(f"Testing on: ckp_{test_ckp_idx} (raw: {raw_ckp})")
                logger.info(f"Accumulated datasets: {len(accumulated_train_data)}")
                
                with self.gpu_manager.context_cleanup():
                    # Train on accumulated data
                    combined_train_data = self._combine_datasets(accumulated_train_data)
                    logger.info(f"Total training samples: {len(combined_train_data) if combined_train_data else 0}")
                    
                    model = self._train_model_on_data(
                        model, combined_train_data, 
                        training_phase=f"accum_ckp_{ckp_idx}_train_0_to_{ckp_idx-1}"
                    )
                    
                    # Test on current checkpoint
                    eval_data = eval_datasets[raw_ckp]
                    eval_results = self._evaluate_model(model, eval_data)
                    
                    checkpoint_results = {
                        'mode': 'accumulative',
                        'train_checkpoints': train_ckp_range,
                        'test_checkpoint': test_ckp_idx,
                        'raw_test_checkpoint': raw_ckp,
                        'evaluation': eval_results,
                        'train_samples': len(combined_train_data) if combined_train_data else 0,
                        'eval_samples': len(eval_data) if eval_data else 0,
                    }
                    
                    results[f'ckp_{ckp_idx}'] = checkpoint_results
                
                # Add current checkpoint's training data for next iteration
                current_train_data = train_datasets[raw_ckp]
                if current_train_data:
                    accumulated_train_data.append(current_train_data)
            
            logger.info(f"Completed checkpoint {ckp_idx} (accumulative)")
        
        return results
    
    def _combine_datasets(self, dataset_list):
        """Combine multiple datasets into one."""
        if not dataset_list:
            return None
        # Placeholder - implement based on your dataset structure
        logger.info(f"Combining {len(dataset_list)} datasets")
        return dataset_list[0]  # Placeholder
    
    def _train_model_on_data(self, model, train_data, training_phase: str):
        """Train model on given data."""
        logger.info(f"Training model: {training_phase}")
        if train_data is None:
            logger.warning("No training data provided")
            return model
        
        # Placeholder - implement actual training
        logger.info(f"Training samples: {len(train_data) if hasattr(train_data, '__len__') else 'unknown'}")
        # Actual training logic will go here
        return model
    
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
