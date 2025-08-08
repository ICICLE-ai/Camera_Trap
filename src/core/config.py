"""
Configuration Management System

Handles loading and merging of configurations from YAML files and command-line arguments.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Main configuration class containing all pipeline parameters."""
    
    # Core experiment settings
    camera: str
    experiment_name: str = "default"
    output_dir: str = "logs"
    seed: int = 42
    device: str = "cuda"
    debug: bool = False
    eval_only: bool = False
    no_save: bool = False
    gpu_cleanup: bool = False
    
    # Model settings
    model: str = "bioclip"
    pretrained_path: Optional[str] = None
    use_peft: bool = False
    peft_config: Dict[str, Any] = None
    
    # Training settings
    epochs: int = 30
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    batch_size: int = 128
    eval_batch_size: int = 512
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    scheduler_params: Dict[str, Any] = None
    
    # Loss settings
    loss_type: str = "ce"
    loss_params: Dict[str, Any] = None
    
    # Data settings
    class_names: List[str] = None
    train_data_path: str = None
    eval_data_path: str = None
    
    # Module settings
    ood_method: str = "all"
    ood_params: Dict[str, Any] = None
    al_method: str = "all"  
    al_params: Dict[str, Any] = None
    cl_method: str = "naive-ft"
    cl_params: Dict[str, Any] = None
    calibration: bool = False
    calibration_params: Dict[str, Any] = None
    
    # Pretrain settings
    pretrain: bool = False
    pretrain_data_path: Optional[str] = None
    pretrain_epochs: int = 10
    
    def __post_init__(self):
        """Initialize default values for nested dictionaries."""
        if self.peft_config is None:
            self.peft_config = {}
        if self.scheduler_params is None:
            self.scheduler_params = {"T_max": self.epochs, "eta_min": 1e-5}
        if self.loss_params is None:
            self.loss_params = {}
        if self.ood_params is None:
            self.ood_params = {}
        if self.al_params is None:
            self.al_params = {}
        if self.cl_params is None:
            self.cl_params = {}
        if self.calibration_params is None:
            self.calibration_params = {}


class ConfigManager:
    """Manages configuration loading and merging."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.configs_dir = self.project_root / "configs"
    
    def load_config(self, args) -> Config:
        """
        Load and merge configuration from multiple sources.
        Priority: command-line args > config file > camera config > defaults
        """
        # Start with default config
        config_dict = {}
        
        # Load camera-specific config if it exists
        camera_config_path = self.configs_dir / "cameras" / f"{args.camera}.yaml"
        if camera_config_path.exists():
            logger.info(f"Loading camera config: {camera_config_path}")
            with open(camera_config_path, 'r') as f:
                camera_config = yaml.safe_load(f)
            config_dict.update(camera_config)
        else:
            logger.warning(f"Camera config not found: {camera_config_path}")
            # Set basic camera info
            config_dict['camera'] = args.camera
            config_dict['class_names'] = self._get_default_class_names(args.camera)
        
        # Load experiment config file if provided
        if args.config:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = self.project_root / config_path
            
            if config_path.exists():
                logger.info(f"Loading experiment config: {config_path}")
                with open(config_path, 'r') as f:
                    exp_config = yaml.safe_load(f)
                config_dict.update(exp_config)
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Override with command-line arguments
        cli_config = self._args_to_dict(args)
        config_dict.update({k: v for k, v in cli_config.items() if v is not None})
        
        # Set output directory if not specified
        if 'output_dir' not in config_dict or not config_dict['output_dir']:
            timestamp = self._get_timestamp()
            exp_name = config_dict.get('experiment_name', 'default')
            if config_dict.get('debug', False):
                config_dict['output_dir'] = f"logs/{args.camera}/debug_{timestamp}"
            else:
                config_dict['output_dir'] = f"logs/{args.camera}/{exp_name}"
        
        # Convert to Config object
        try:
            config = Config(**config_dict)
        except TypeError as e:
            logger.error(f"Error creating config object: {e}")
            logger.error(f"Config dict keys: {list(config_dict.keys())}")
            raise
        
        return config
    
    def _args_to_dict(self, args) -> Dict[str, Any]:
        """Convert command-line arguments to dictionary."""
        return {
            'camera': args.camera,
            'model': args.model,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'ood_method': args.ood_method,
            'al_method': args.al_method, 
            'cl_method': args.cl_method,
            'use_peft': args.use_peft,
            'calibration': args.calibration,
            'device': args.device,
            'seed': args.seed,
            'gpu_cleanup': args.gpu_cleanup,
            'debug': args.debug,
            'eval_only': args.eval_only,
            'no_save': args.no_save,
            'output_dir': args.output_dir,
        }
    
    def _get_default_class_names(self, camera: str) -> List[str]:
        """Get default class names for a camera if not in config."""
        # This is a fallback - ideally class names should be in camera config
        default_classes = [
            "baboon", "buffalo", "duikercommongrey", "elephant", "giraffe",
            "impala", "kudu", "steenbok", "warthog", "wildebeestblue", "zebraburchells"
        ]
        logger.warning(f"Using default class names for {camera}. Consider creating a camera config file.")
        return default_classes
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_config(self, config: Config, output_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
