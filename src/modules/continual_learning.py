"""
Continual Learning Module
"""

import logging
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import Config

logger = logging.getLogger(__name__)


class CLFactory:
    """Factory for creating Continual Learning modules."""
    
    @staticmethod
    def create(method: str, config: Config) -> 'BaseCLModule':
        """Create CL module based on method name."""
        
        method = method.lower()
        
        if method == 'none':
            return CLNone(config)
        elif method == 'naive-ft' or method == 'naive_ft':
            return CLNaiveFT(config)
        elif method == 'accumulative':
            return CLAccumulative(config)
        elif method == 'ewc':
            return CLEWC(config)  # Elastic Weight Consolidation
        elif method == 'replay':
            return CLReplay(config)
        else:
            logger.warning(f"Unknown CL method: {method}, using 'naive-ft'")
            return CLNaiveFT(config)


class BaseCLModule(ABC):
    """Base class for Continual Learning modules."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.buffer = []  # For storing previous samples
    
    @abstractmethod
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """
        Process continual learning training.
        
        Args:
            model: Current model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            al_mask: Active learning mask indicating selected samples
            checkpoint: Current checkpoint name
            
        Returns:
            Updated model after training
        """
        pass
    
    def pretrain(self, model: nn.Module, pretrain_dataset, epochs: int) -> nn.Module:
        """
        Pretrain the model on a pretraining dataset.
        
        Args:
            model: Model to pretrain
            pretrain_dataset: Pretraining dataset
            epochs: Number of pretraining epochs
            
        Returns:
            Pretrained model
        """
        logger.info(f"Pretraining model for {epochs} epochs")
        
        if pretrain_dataset is None or len(pretrain_dataset) == 0:
            logger.warning("No pretraining dataset provided, skipping pretraining")
            return model
        
        # Create dataloader
        pretrain_loader = self._create_dataloader(pretrain_dataset, shuffle=True)
        
        # Train model
        trained_model = self._train_model(model, pretrain_loader, epochs, "Pretrain")
        
        return trained_model
    
    def _train_model(self, model: nn.Module, dataloader, epochs: int, 
                     phase_name: str = "Train") -> nn.Module:
        """Train model using the provided dataloader."""
        
        if dataloader is None or len(dataloader) == 0:
            logger.warning(f"{phase_name}: No data to train on, skipping training")
            return model
        
        logger.info(f"{phase_name}: Training for {epochs} epochs on {len(dataloader.dataset)} samples")
        
        # Setup optimizer
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer) if self.config.scheduler else None
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self._compute_loss(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log progress
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            if epoch % max(1, epochs // 5) == 0:  # Log every 20% of epochs
                logger.info(f"{phase_name} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"{phase_name}: Training completed")
        return model
    
    def _create_dataloader(self, dataset, shuffle: bool = True):
        """Create a dataloader for the dataset."""
        
        if dataset is None:
            return None
        
        # Placeholder implementation - would use actual DataLoader
        # For now, return a simple mock dataloader
        class MockDataLoader:
            def __init__(self, dataset, batch_size, shuffle):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
            
            def __iter__(self):
                # Placeholder - return empty iterator
                return iter([])
            
            def __len__(self):
                return 0
        
        return MockDataLoader(dataset, self.config.batch_size, shuffle)
    
    def _create_optimizer(self, model: nn.Module):
        """Create optimizer for training."""
        
        optimizer_name = self.config.optimizer.lower()
        
        if optimizer_name == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using AdamW")
            return optim.AdamW(model.parameters(), lr=self.config.learning_rate)
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        
        scheduler_name = self.config.scheduler.lower()
        
        if scheduler_name == 'cosineannealinglr':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_params.get('T_max', self.config.epochs),
                eta_min=self.config.scheduler_params.get('eta_min', 1e-5)
            )
        elif scheduler_name == 'steplr':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_params.get('step_size', 10),
                gamma=self.config.scheduler_params.get('gamma', 0.1)
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}")
            return None
    
    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        
        loss_type = self.config.loss_type.lower()
        
        if loss_type == 'ce':
            return nn.functional.cross_entropy(outputs, labels)
        elif loss_type == 'focal':
            # Placeholder for focal loss implementation
            return nn.functional.cross_entropy(outputs, labels)
        else:
            return nn.functional.cross_entropy(outputs, labels)


class CLNone(BaseCLModule):
    """No continual learning - just return the model as-is."""
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """Return model without training."""
        
        logger.info("CL: No training - zero-shot evaluation")
        return model


class CLNaiveFT(BaseCLModule):
    """Naive fine-tuning on selected samples."""
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """Fine-tune model on selected samples."""
        
        logger.info("CL: Naive fine-tuning")
        
        # Get selected samples based on AL mask
        selected_dataset = self._filter_dataset(train_dataset, al_mask)
        
        # Create dataloader
        train_loader = self._create_dataloader(selected_dataset, shuffle=True)
        
        # Train model
        trained_model = self._train_model(model, train_loader, self.config.epochs, "Naive-FT")
        
        return trained_model
    
    def _filter_dataset(self, dataset, mask: np.ndarray):
        """Filter dataset based on boolean mask."""
        # Placeholder implementation
        return dataset


class CLAccumulative(BaseCLModule):
    """Accumulative training on all samples seen so far."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.accumulated_samples = []
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """Train on all accumulated samples."""
        
        logger.info("CL: Accumulative training")
        
        # Add new selected samples to buffer
        selected_samples = self._filter_dataset(train_dataset, al_mask)
        self.accumulated_samples.extend(selected_samples)
        
        logger.info(f"Accumulated {len(self.accumulated_samples)} samples total")
        
        # Create dataloader with all accumulated samples
        train_loader = self._create_dataloader(self.accumulated_samples, shuffle=True)
        
        # Train model
        trained_model = self._train_model(model, train_loader, self.config.epochs, "Accumulative")
        
        return trained_model
    
    def _filter_dataset(self, dataset, mask: np.ndarray):
        """Filter dataset and return list of samples."""
        # Placeholder implementation
        return [] if dataset is None else []


class CLEWC(BaseCLModule):
    """Elastic Weight Consolidation for continual learning."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.fisher_info = None
        self.optimal_params = None
        self.ewc_lambda = config.cl_params.get('ewc_lambda', 0.4)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """Train with EWC regularization."""
        
        logger.info(f"CL: EWC training (lambda={self.ewc_lambda})")
        
        # Compute Fisher Information Matrix if we have previous task
        if self.optimal_params is not None:
            logger.info("Computing Fisher Information Matrix...")
            self._compute_fisher_info(model, eval_dataset)
        
        # Get selected samples
        selected_dataset = self._filter_dataset(train_dataset, al_mask)
        train_loader = self._create_dataloader(selected_dataset, shuffle=True)
        
        # Train with EWC penalty
        trained_model = self._train_with_ewc(model, train_loader)
        
        # Store current parameters as optimal for next task
        self.optimal_params = {name: param.clone().detach() 
                             for name, param in model.named_parameters()}
        
        return trained_model
    
    def _compute_fisher_info(self, model: nn.Module, dataset):
        """Compute Fisher Information Matrix."""
        # Placeholder implementation
        self.fisher_info = {name: torch.zeros_like(param) 
                          for name, param in model.named_parameters()}
    
    def _train_with_ewc(self, model: nn.Module, dataloader) -> nn.Module:
        """Train model with EWC regularization."""
        # Placeholder implementation - would implement EWC training loop
        return self._train_model(model, dataloader, self.config.epochs, "EWC")
    
    def _filter_dataset(self, dataset, mask: np.ndarray):
        """Filter dataset based on mask."""
        return dataset


class CLReplay(BaseCLModule):
    """Experience replay for continual learning."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer_size = config.cl_params.get('buffer_size', 1000)
        self.replay_ratio = config.cl_params.get('replay_ratio', 0.2)
    
    def process(self, model: nn.Module, train_dataset, eval_dataset,
                al_mask: np.ndarray, checkpoint: str) -> nn.Module:
        """Train with experience replay."""
        
        logger.info(f"CL: Replay training (buffer_size={self.buffer_size})")
        
        # Get selected samples
        selected_samples = self._filter_dataset(train_dataset, al_mask)
        
        # Update replay buffer
        self._update_buffer(selected_samples)
        
        # Create mixed dataset (new samples + replay samples)
        mixed_dataset = self._create_mixed_dataset(selected_samples)
        train_loader = self._create_dataloader(mixed_dataset, shuffle=True)
        
        # Train model
        trained_model = self._train_model(model, train_loader, self.config.epochs, "Replay")
        
        return trained_model
    
    def _update_buffer(self, new_samples):
        """Update experience replay buffer."""
        # Add new samples to buffer
        self.buffer.extend(new_samples)
        
        # Maintain buffer size limit
        if len(self.buffer) > self.buffer_size:
            # Random removal to maintain diversity
            indices_to_keep = np.random.choice(
                len(self.buffer), 
                size=self.buffer_size, 
                replace=False
            )
            self.buffer = [self.buffer[i] for i in indices_to_keep]
        
        logger.info(f"Replay buffer updated: {len(self.buffer)} samples")
    
    def _create_mixed_dataset(self, new_samples):
        """Create dataset mixing new samples with replay samples."""
        
        n_replay = int(len(new_samples) * self.replay_ratio)
        n_replay = min(n_replay, len(self.buffer))
        
        if n_replay > 0:
            replay_samples = np.random.choice(self.buffer, size=n_replay, replace=False)
            mixed_samples = list(new_samples) + list(replay_samples)
        else:
            mixed_samples = new_samples
        
        logger.info(f"Mixed dataset: {len(new_samples)} new + {n_replay} replay = {len(mixed_samples)} total")
        
        return mixed_samples
    
    def _filter_dataset(self, dataset, mask: np.ndarray):
        """Filter dataset based on mask."""
        return [] if dataset is None else []
