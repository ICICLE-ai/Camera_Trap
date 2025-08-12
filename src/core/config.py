"""
Configuration Management System

Handles loading and merging of configurations from YAML files and command-line arguments.
Auto-parses camera names and detects data structure.
"""

import os
import yaml
import json
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
    dataset: str = None  # Auto-parsed from camera
    camera_name: str = None  # Auto-parsed from camera
    day_interval: int = 30
    experiment_name: str = "default"
    output_dir: str = "logs"
    seed: int = 42
    device: str = "cuda"
    debug: bool = False
    eval_only: bool = False
    no_save: bool = False
    gpu_cleanup: bool = False
    
    # Training mode settings
    mode: str = "accum"  # Training mode: "zs" (zero-shot), "oracle" (all data), "accum" (continual learning)
    
    # Auto-detected paths and classes
    data_root: str = "data"
    train_data_path: str = None
    eval_data_path: str = None
    class_names: List[str] = None
    checkpoint_info: Dict[str, Any] = None  # Contains parsed checkpoint data
    
    # Dataset processing settings
    class_field: str = "common"  # Which field to use: "common", "query", "scientific"
    min_conf: float = 0.5
    max_samples_per_class: Optional[int] = None
    filter_low_conf: bool = True
    remove_duplicates: bool = True
    balance_classes: bool = False
    checkpoint_prefix: str = "ckp_"
    start_checkpoint: int = 0
    end_checkpoint: Optional[int] = None
    
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
        Priority: command-line args > training config > dataset config > defaults
        """
        # Start with default config
        config_dict = {}
        
        # Load dataset configuration first
        dataset_config = self._load_dataset_config()
        config_dict.update(dataset_config)
        
        # Parse camera name to extract dataset and camera components
        dataset, camera_name, full_camera_name = self._parse_camera_name(args.camera)
        config_dict.update({
            'camera': args.camera,
            'dataset': dataset,
            'camera_name': camera_name,
        })
        
        # Auto-detect data paths and class names
        data_info = self._detect_data_structure(
            dataset, full_camera_name,  # Use full camera name for directory path
            config_dict.get('day_interval', 30),
            config_dict.get('class_field', 'common'),
            config_dict.get('min_conf', 0.5),
            config_dict.get('filter_low_conf', True)
        )
        config_dict.update(data_info)
        
        # Load training config file if provided
        if args.config:
            training_config = self._load_training_config(args.config)
            config_dict.update(training_config)
        
        # Override with command-line arguments (highest priority)
        cli_config = self._args_to_dict(args)
        config_dict.update({k: v for k, v in cli_config.items() if v is not None})
        
        # Set output directory if not specified
        if 'output_dir' not in config_dict or not config_dict['output_dir']:
            timestamp = self._get_timestamp()
            exp_name = config_dict.get('experiment_name', 'default')
            if config_dict.get('debug', False):
                config_dict['output_dir'] = f"logs/{dataset}_{camera_name}/debug_{timestamp}"
            else:
                config_dict['output_dir'] = f"logs/{dataset}_{camera_name}/{exp_name}"
        
        # Convert to Config object
        try:
            # Filter out fields that are not part of the Config dataclass
            from dataclasses import fields
            config_fields = {field.name for field in fields(Config)}
            filtered_config = {k: v for k, v in config_dict.items() if k in config_fields}
            
            # Log any filtered fields for debugging
            filtered_out = set(config_dict.keys()) - config_fields
            if filtered_out:
                logger.debug(f"Filtered out non-Config fields: {sorted(filtered_out)}")
            
            config = Config(**filtered_config)
        except TypeError as e:
            logger.error(f"Error creating config object: {e}")
            logger.error(f"Config dict keys: {list(config_dict.keys())}")
            raise
        
        return config
    
    def _load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration from configs/dataset.yaml"""
        dataset_config_path = self.configs_dir / "dataset.yaml"
        
        if dataset_config_path.exists():
            logger.info(f"Loading dataset config: {dataset_config_path}")
            with open(dataset_config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            logger.warning(f"Dataset config not found: {dataset_config_path}")
            return {}
    
    def _load_training_config(self, config_name: str) -> Dict[str, Any]:
        """Load training configuration from configs/training/"""
        config_path = Path(config_name)
        
        # If not absolute path, look in configs/training/
        if not config_path.is_absolute():
            if not config_path.name.endswith('.yaml'):
                config_path = config_path.with_suffix('.yaml')
            config_path = self.configs_dir / "training" / config_path.name
        
        if config_path.exists():
            logger.info(f"Loading training config: {config_path}")
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Training config not found: {config_path}")
    
    
    def _parse_camera_name(self, camera_input: str) -> tuple[str, str, str]:
        """
        Parse camera input string to extract dataset, camera name, and full camera directory.
        
        Args:
            camera_input: Input like 'APN_K024'
            
        Returns:
            Tuple of (dataset, camera_short_name, full_camera_name) like ('APN', 'K024', 'APN_K024')
        """
        parts = camera_input.split('_', 1)
        
        if len(parts) >= 2:
            dataset = parts[0]
            camera_short_name = parts[1]
            full_camera_name = camera_input  # Use full name for directory
        else:
            # If no underscore, treat the whole thing as camera name
            dataset = "unknown"
            camera_short_name = camera_input
            full_camera_name = camera_input
            logger.warning(f"Cannot parse dataset from '{camera_input}', using 'unknown' as dataset")
        
        logger.info(f"Parsed camera: '{camera_input}' -> dataset='{dataset}', camera='{camera_short_name}', full_dir='{full_camera_name}'")
        return dataset, camera_short_name, full_camera_name
    
    def _detect_data_structure(self, dataset: str, full_camera_name: str, day_interval: int, 
                             class_field: str, min_conf: float, filter_low_conf: bool) -> Dict[str, Any]:
        """
        Detect data structure and extract class names from JSON files.
        
        Expected structure: data/{dataset}/{full_camera_name}/{day_interval}/train.json
                           data/{dataset}/{full_camera_name}/{day_interval}/test.json
        """
        data_root = self.project_root / "data"
        data_dir = data_root / dataset / full_camera_name / str(day_interval)
        
        train_path = data_dir / "train.json"
        test_path = data_dir / "test.json"
        
        # Check if data directory exists
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please organize your data as: data/{dataset}/{full_camera_name}/{day_interval}/\n"
                f"Available day_interval options: {self._get_available_day_intervals(data_root / dataset / full_camera_name)}"
            )
        
        # Check if JSON files exist
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        logger.info(f"Found data directory: {data_dir}")
        logger.info(f"Train data: {train_path}")
        logger.info(f"Test data: {test_path}")
        
        # Extract class names and checkpoint info from JSON files
        class_names, checkpoint_info = self._extract_class_names_and_checkpoints(
            train_path, test_path, class_field, min_conf, filter_low_conf
        )
        
        result = {
            'data_root': str(data_root),
            'train_data_path': str(train_path),
            'eval_data_path': str(test_path),
            'class_names': class_names,
            'checkpoint_info': checkpoint_info
        }
        
        logger.info(f"Dataset info: {len(class_names)} classes, {len(checkpoint_info.get('train', {}))} train checkpoints")
        
        return result
    
    def _get_available_day_intervals(self, camera_dir: Path) -> List[str]:
        """Get available day intervals for a camera."""
        if not camera_dir.exists():
            return []
        
        intervals = []
        for item in camera_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                intervals.append(item.name)
        
        return sorted(intervals, key=int) if intervals else []
    
    def _extract_class_names_and_checkpoints(self, train_path: Path, test_path: Path, 
                                            class_field: str, min_conf: float, 
                                            filter_low_conf: bool) -> tuple[List[str], Dict[str, Any]]:
        """Extract unique class names and checkpoint information from train and test JSON files."""
        
        class_names_set = set()
        checkpoint_info = {'train': {}, 'test': {}}
        
        # Process train.json
        try:
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            
            train_classes, train_ckpts = self._process_checkpoint_json(
                train_data, "train", class_field, min_conf, filter_low_conf
            )
            class_names_set.update(train_classes)
            checkpoint_info['train'] = train_ckpts
            
        except Exception as e:
            logger.error(f"Error reading train.json: {e}")
            raise
        
        # Process test.json
        try:
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            
            test_classes, test_ckpts = self._process_checkpoint_json(
                test_data, "test", class_field, min_conf, filter_low_conf
            )
            class_names_set.update(test_classes)
            checkpoint_info['test'] = test_ckpts
            
        except Exception as e:
            logger.error(f"Error reading test.json: {e}")
            raise
        
        # Convert to sorted list
        class_names = sorted(list(class_names_set))
        
        logger.info(f"Extracted {len(class_names)} unique classes using '{class_field}' field: {class_names}")
        logger.info(f"Found {len(checkpoint_info['train'])} train checkpoints and {len(checkpoint_info['test'])} test checkpoints")
        
        return class_names, checkpoint_info
    
    def _process_checkpoint_json(self, data: Dict, data_type: str, class_field: str, 
                               min_conf: float, filter_low_conf: bool) -> tuple[List[str], Dict[str, Any]]:
        """
        Process JSON data with checkpoint structure or simple list format.
        
        Supports two formats:
        1. Checkpoint format: {"ckp_0": [...], "ckp_1": [...], ...}
        2. Simple list format: [{"image_path": "...", "label": "...", ...}, ...]
        """
        classes = set()
        checkpoint_data = {}
        
        # Handle simple list format
        if isinstance(data, list):
            logger.info(f"Processing {data_type}.json as simple list format")
            return self._process_simple_list_json(data, data_type, class_field, min_conf, filter_low_conf)
        
        # Handle checkpoint format
        elif isinstance(data, dict):
            # Check if it's checkpoint format
            checkpoint_keys = [k for k in data.keys() if k.startswith("ckp_")]
            if checkpoint_keys:
                logger.info(f"Processing {data_type}.json as checkpoint format with {len(checkpoint_keys)} checkpoints")
                return self._process_checkpoint_format(data, data_type, class_field, min_conf, filter_low_conf)
            else:
                # Try to handle as simple dict format
                logger.info(f"Processing {data_type}.json as simple dict format")
                return self._process_simple_dict_json(data, data_type, class_field, min_conf, filter_low_conf)
        
        else:
            raise ValueError(f"Unsupported JSON format for {data_type}.json: {type(data)}")
    
    def _process_simple_list_json(self, data: List, data_type: str, class_field: str, 
                                min_conf: float, filter_low_conf: bool) -> tuple[List[str], Dict[str, Any]]:
        """Process simple list format JSON."""
        classes = set()
        
        # Filter and process samples
        filtered_samples = []
        
        for sample in data:
            if not isinstance(sample, dict):
                continue
            
            # Check required fields
            if 'image_path' not in sample:
                logger.warning(f"Sample missing 'image_path' field, skipping")
                continue
            
            # Try different common field names if specified field not found
            class_value = None
            if class_field in sample:
                class_value = sample[class_field]
            else:
                # Try common field names as fallback
                common_fields = ['label', 'class', 'category', 'common', 'scientific']
                for field in common_fields:
                    if field in sample and sample[field]:
                        class_value = sample[field]
                        if class_field != field:
                            logger.warning(f"Using '{field}' instead of '{class_field}' for class name")
                        break
            
            if not class_value:
                logger.warning(f"Sample missing valid class field, skipping. Available fields: {list(sample.keys())}")
                continue
            
            # Apply confidence filtering
            sample_conf = sample.get('conf', 1.0)  # Default confidence of 1.0 if not specified
            if filter_low_conf and sample_conf < min_conf:
                continue
            
            classes.add(class_value)
            filtered_samples.append(sample)
        
        # Create single checkpoint entry for simple format
        checkpoint_data = {
            'ckp_0': {
                'samples': filtered_samples,
                'num_samples': len(filtered_samples),
                'classes': sorted(list(classes)),
                'num_classes': len(classes)
            }
        }
        
        logger.info(f"{data_type}: {len(filtered_samples)} samples, {len(classes)} classes")
        
        if not classes:
            logger.error(f"No valid classes found in {data_type} data")
            raise ValueError(f"No valid classes extracted from {data_type}.json")
        
        return list(classes), checkpoint_data
    
    def _process_checkpoint_format(self, data: Dict, data_type: str, class_field: str, 
                                 min_conf: float, filter_low_conf: bool) -> tuple[List[str], Dict[str, Any]]:
        """Process checkpoint format JSON."""
        classes = set()
        checkpoint_data = {}
        
        # Process each checkpoint
        for checkpoint_key, samples in data.items():
            if not checkpoint_key.startswith("ckp_"):
                logger.warning(f"Skipping non-checkpoint key: {checkpoint_key}")
                continue
            
            if not isinstance(samples, list):
                logger.warning(f"Skipping {checkpoint_key}: expected list, got {type(samples)}")
                continue
            
            # Filter and process samples
            filtered_samples = []
            checkpoint_classes = set()
            
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                
                # Check required fields
                if 'image_path' not in sample:
                    logger.warning(f"Sample missing 'image_path' field, skipping")
                    continue
                
                if class_field not in sample:
                    logger.warning(f"Sample missing '{class_field}' field, skipping")
                    continue
                
                # Apply confidence filtering
                sample_conf = sample.get('conf', 1.0)
                if filter_low_conf and sample_conf < min_conf:
                    continue
                
                # Extract class name
                class_name = sample[class_field]
                if class_name:  # Skip empty class names
                    classes.add(class_name)
                    checkpoint_classes.add(class_name)
                    filtered_samples.append(sample)
            
            # Store checkpoint info
            checkpoint_data[checkpoint_key] = {
                'samples': filtered_samples,
                'num_samples': len(filtered_samples),
                'classes': sorted(list(checkpoint_classes)),
                'num_classes': len(checkpoint_classes)
            }
            
            logger.debug(f"{data_type} {checkpoint_key}: {len(filtered_samples)} samples, "
                        f"{len(checkpoint_classes)} classes")
        
        if not classes:
            logger.error(f"No valid classes found in {data_type} data using field '{class_field}'")
            logger.error("Please check:")
            logger.error(f"1. The class field name (current: '{class_field}')")
            logger.error(f"2. Confidence threshold (current: {min_conf})")
            logger.error("3. JSON structure matches expected format")
            
            # Show sample of available fields
            if checkpoint_data:
                sample_ckp = next(iter(checkpoint_data.values()))
                if sample_ckp['samples']:
                    sample_fields = list(sample_ckp['samples'][0].keys())
                    logger.error(f"Available fields in samples: {sample_fields}")
            
            raise ValueError(f"No valid classes extracted from {data_type}.json")
        
        return list(classes), checkpoint_data
    
    def _process_simple_dict_json(self, data: Dict, data_type: str, class_field: str, 
                                min_conf: float, filter_low_conf: bool) -> tuple[List[str], Dict[str, Any]]:
        """Process simple dict format (fallback for other dict structures)."""
        classes = set()
        
        # Try to find samples in common dict structures
        samples = []
        if 'samples' in data:
            samples = data['samples']
        elif 'images' in data:
            samples = data['images']
        elif 'data' in data:
            samples = data['data']
        else:
            # Treat the whole dict as containing sample entries
            for key, value in data.items():
                if isinstance(value, list):
                    samples = value
                    break
        
        if not samples:
            logger.warning(f"No samples found in {data_type}.json dict structure")
            return [], {'ckp_0': {'samples': [], 'num_samples': 0, 'classes': [], 'num_classes': 0}}
        
        # Process as simple list
        return self._process_simple_list_json(samples, data_type, class_field, min_conf, filter_low_conf)
    
    def _args_to_dict(self, args) -> Dict[str, Any]:
        """Convert command line arguments to dictionary for config merging."""
        
        args_dict = {}
        
        # Basic configuration
        if hasattr(args, 'config') and args.config:
            args_dict['training_config_path'] = args.config
        
        if hasattr(args, 'dataset_config') and args.dataset_config:
            args_dict['dataset_config_path'] = args.dataset_config
            
        # Experiment settings
        if hasattr(args, 'experiment_name') and args.experiment_name:
            args_dict['experiment_name'] = args.experiment_name
            
        if hasattr(args, 'output_dir') and args.output_dir:
            args_dict['output_dir'] = args.output_dir
        
        # Data configuration
        if hasattr(args, 'data_dir') and args.data_dir:
            args_dict['data_dir'] = args.data_dir
            
        if hasattr(args, 'dataset') and args.dataset:
            args_dict['dataset'] = args.dataset
            
        if hasattr(args, 'camera') and args.camera:
            args_dict['camera'] = args.camera
            
        if hasattr(args, 'day_interval') and args.day_interval:
            args_dict['day_interval'] = args.day_interval
        
        # Dataset-specific parameters
        if hasattr(args, 'class_field') and args.class_field:
            args_dict['class_field'] = args.class_field
            
        if hasattr(args, 'min_conf') and args.min_conf is not None:
            args_dict['min_conf'] = args.min_conf
            
        if hasattr(args, 'filter_low_conf') and args.filter_low_conf is not None:
            args_dict['filter_low_conf'] = args.filter_low_conf
            
        if hasattr(args, 'max_samples_per_class') and args.max_samples_per_class is not None:
            args_dict['max_samples_per_class'] = args.max_samples_per_class
            
        if hasattr(args, 'balance_classes') and args.balance_classes is not None:
            args_dict['balance_classes'] = args.balance_classes
            
        if hasattr(args, 'train_val_split') and args.train_val_split is not None:
            args_dict['train_val_split'] = args.train_val_split
            
        if hasattr(args, 'use_stratify') and args.use_stratify is not None:
            args_dict['use_stratify'] = args.use_stratify
        
        # Model configuration (handle both old and new arg names)
        if hasattr(args, 'model_name') and args.model_name:
            args_dict['model_name'] = args.model_name
        elif hasattr(args, 'model') and args.model:
            args_dict['model_name'] = args.model
            
        if hasattr(args, 'pretrained') and args.pretrained is not None:
            args_dict['pretrained'] = args.pretrained
        
        # Training configuration (handle both old and new arg names)
        if hasattr(args, 'epochs') and args.epochs is not None:
            args_dict['epochs'] = args.epochs
            
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            args_dict['batch_size'] = args.batch_size
            
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            args_dict['learning_rate'] = args.learning_rate
        elif hasattr(args, 'lr') and args.lr is not None:
            args_dict['learning_rate'] = args.lr
        
        # Module configuration (handle both old and new styles)
        if hasattr(args, 'use_ood') and args.use_ood is not None:
            args_dict['use_ood'] = args.use_ood
        elif hasattr(args, 'ood_method') and args.ood_method:
            args_dict['ood_method'] = args.ood_method
            
        if hasattr(args, 'use_al') and args.use_al is not None:
            args_dict['use_al'] = args.use_al
        elif hasattr(args, 'al_method') and args.al_method:
            args_dict['al_method'] = args.al_method
            
        if hasattr(args, 'use_cl') and args.use_cl is not None:
            args_dict['use_cl'] = args.use_cl
        elif hasattr(args, 'cl_method') and args.cl_method:
            args_dict['cl_method'] = args.cl_method
            
        if hasattr(args, 'use_peft') and args.use_peft is not None:
            args_dict['use_peft'] = args.use_peft
            
        if hasattr(args, 'use_calibration') and args.use_calibration is not None:
            args_dict['use_calibration'] = args.use_calibration
        elif hasattr(args, 'calibration') and args.calibration is not None:
            args_dict['use_calibration'] = args.calibration
        
        # System configuration
        if hasattr(args, 'device') and args.device:
            args_dict['device'] = args.device
            
        if hasattr(args, 'mode') and args.mode:
            args_dict['mode'] = args.mode
            
        if hasattr(args, 'gpu_cleanup') and args.gpu_cleanup is not None:
            args_dict['gpu_cleanup'] = args.gpu_cleanup
            
        if hasattr(args, 'verbose') and args.verbose is not None:
            args_dict['verbose'] = args.verbose
        elif hasattr(args, 'debug') and args.debug is not None:
            args_dict['verbose'] = args.debug
            
        if hasattr(args, 'seed') and args.seed is not None:
            args_dict['seed'] = args.seed
        
        # Additional old arguments for backward compatibility
        if hasattr(args, 'eval_only') and args.eval_only is not None:
            args_dict['eval_only'] = args.eval_only
            
        if hasattr(args, 'no_save') and args.no_save is not None:
            args_dict['no_save'] = args.no_save
        
        return args_dict
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_config(self, config: Config, output_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
