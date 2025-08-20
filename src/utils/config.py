"""
Configuration management utilities
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def update_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Parsed command line arguments
        
    Returns:
        Updated configuration
    """
    updated_config = config.copy()
    
    # Map command line arguments to config keys
    arg_mappings = {
        'camera': 'camera',
        'mode': 'mode',
        'device': 'device', 
        'epochs': 'training.epochs',
        'batch_size': 'training.batch_size',
        'lr': 'training.learning_rate',
        'seed': 'seed',
        'debug': 'debug',
        'model_version': 'model_version',
        'use_peft': 'use_peft',
        'gpu_cleanup': 'gpu_cleanup',
        'no_save': 'no_save',
        'eval_only': 'eval_only',
        'calibration': 'calibration',
        'output_dir': 'output_dir',
        'model_path': 'model_path'
    }
    
    # Update config with provided arguments
    for arg_name, config_key in arg_mappings.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                # Handle nested config keys
                keys = config_key.split('.')
                target = updated_config
                for key in keys[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                target[keys[-1]] = arg_value
    
    return updated_config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {output_path}: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'camera',
        'mode',
        'device',
        'seed'
    ]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required configuration field: {field}")
            return False
    
    # Validate nested training config if present
    if 'training' in config:
        training_required = ['batch_size', 'learning_rate', 'epochs']
        for field in training_required:
            if field not in config['training']:
                # logger.warning(f"Missing training configuration field: {field}")
                pass
    
    # logger.info("Configuration validation passed")
    return True


def get_mode_type(config_path: str) -> str:
    """
    Extract mode type from config path.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Mode type string
    """
    path = Path(config_path)
    
    mode_mapping = {
        'zs.yaml': 'zs',
        'oracle.yaml': 'oracle', 
        'accumulative.yaml': 'accumulative'
    }
    
    return mode_mapping.get(path.name, path.stem)


def create_experiment_config(base_config: Dict[str, Any], 
                           experiment_name: str = None) -> Dict[str, Any]:
    """
    Create experiment configuration with metadata.
    
    Args:
        base_config: Base configuration
        experiment_name: Name of the experiment
        
    Returns:
        Experiment configuration with metadata
    """
    experiment_config = base_config.copy()
    
    # Add experiment metadata
    experiment_config['experiment'] = {
        'name': experiment_name or 'unnamed_experiment',
        'timestamp': datetime.now().isoformat(),
        'config_version': '2.0'
    }
    
    # Add system information
    import platform
    import torch
    
    experiment_config['system'] = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    return experiment_config


def get_config_summary(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get a summary of key configuration parameters for logging.
    
    Args:
        config: Full configuration
        
    Returns:
        Summary dictionary with key parameters
    """
    summary = {}
    
    # Core parameters
    summary['Camera'] = config.get('camera', 'Unknown')
    summary['Mode'] = config.get('mode', 'Unknown')
    summary['Device'] = config.get('device', 'Unknown')
    summary['Seed'] = str(config.get('seed', 'Unknown'))
    
    # Model parameters
    if 'model' in config:
        model_config = config['model']
        summary['Model'] = model_config.get('name', 'Unknown')
        summary['Model Version'] = config.get('model_version', 'v1')
        summary['Pretrained'] = str(model_config.get('pretrained', False))
        summary['Use PEFT'] = str(model_config.get('use_peft', False))
    
    # Training parameters
    if 'training' in config:
        training_config = config['training']
        summary['Epochs'] = str(training_config.get('epochs', 0))
        summary['Batch Size'] = str(training_config.get('batch_size', 'Unknown'))
        summary['Learning Rate'] = str(training_config.get('learning_rate', 'Unknown'))
        summary['Optimizer'] = training_config.get('optimizer', 'Unknown')
    
    return summary
