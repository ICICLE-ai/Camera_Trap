"""Configuration management for camera trap detection."""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Configuration manager for hierarchical YAML configs."""
    
    def __init__(self, config_path: str):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with inheritance support."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if 'inherit_from' in config:
            parent_path = self.config_path.parent / config['inherit_from']
            parent_config = ConfigManager(parent_path).get_config()
            config = self._merge_configs(parent_config, config)
            del config['inherit_from']  # Remove inheritance key
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def validate(self) -> bool:
        """Validate configuration structure."""
        required_sections = ['experiment', 'model', 'data', 'training']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate experiment section
        exp_config = self.config['experiment']
        if 'name' not in exp_config:
            raise ValueError("Missing experiment name")
        
        # Validate model section
        model_config = self.config['model']
        if 'name' not in model_config:
            raise ValueError("Missing model name")
        
        # Validate data section
        data_config = self.config['data']
        if 'train_path' not in data_config:
            raise ValueError("Missing training data path")
        
        # Validate training section
        training_config = self.config['training']
        required_training_keys = ['epochs', 'batch_size', 'learning_rate']
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing training parameter: {key}")
        
        return True
    
    def update(self, key: str, value: Any) -> None:
        """Update a configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
