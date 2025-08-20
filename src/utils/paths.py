"""
Path and directory management utilities
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Tuple
import logging
import json

logger = logging.getLogger(__name__)


def setup_experiment_directories(camera: str, mode_type: str, base_dir: str = "logs") -> Tuple[str, str]:
    """
    Setup experiment directories with consistent structure.
    
    Args:
        camera: Camera identifier (e.g., 'APN_K024')
        mode_type: Mode type (e.g., 'zs', 'oracle', 'accumulative') 
        base_dir: Base directory for logs
        
    Returns:
        Tuple of (log_directory, timestamp)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Extract camera group and specific camera
    camera_parts = camera.split('_')
    camera_group = camera_parts[0] if camera_parts else camera
    
    # Create hierarchical directory structure
    # logs/APN/APN_K024/30/zs/2025-08-14-16-38-31/
    log_dir = os.path.join(
        base_dir,
        camera_group,
        camera,
        "30",  # Default epoch value - could be parameterized
        mode_type,
        timestamp
    )
    
    # Create directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create directory (silent operation)
    # logger.info(f"Created experiment directory: {log_dir}")
    return log_dir, timestamp


def load_checkpoint_data(file_path: str) -> dict:
    """
    Load checkpoint data from JSON file.
    
    Args:
        file_path: Path to JSON file containing checkpoint data
        
    Returns:
        Dictionary with checkpoint data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load checkpoint data from {file_path}: {e}")
        return {}


def get_checkpoint_directories(camera: str, base_data_dir: str = "data") -> list:
    """
    Get list of checkpoint names from test data JSON file.
    
    Args:
        camera: Camera identifier
        base_data_dir: Base data directory
        
    Returns:
        Sorted list of checkpoint names (e.g., ['ckp_1', 'ckp_2', ...])
    """
    # Handle camera path structure: data/APN/APN_K024/30/test.json
    camera_parts = camera.split('_')
    if len(camera_parts) >= 2:
        camera_group = camera_parts[0]  # e.g., 'APN'
        test_file = os.path.join(base_data_dir, camera_group, camera, "30", "test.json")
    else:
        test_file = os.path.join(base_data_dir, camera, "30", "test.json")
    
    if not os.path.exists(test_file):
        logger.warning(f"Test data file not found: {test_file}")
        return []
    
    # Load test data and extract checkpoint keys
    test_data = load_checkpoint_data(test_file)
    
    # Get all checkpoint keys (ckp_1, ckp_2, etc.)
    checkpoints = [key for key in test_data.keys() if key.startswith('ckp_')]
    
    # Sort numerically (ckp_1, ckp_2, ..., ckp_10, ...)
    def sort_key(ckp_name):
        try:
            return int(ckp_name.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=sort_key)
    # logger.info(f"Found {len(checkpoints)} checkpoints for camera {camera}")
    return checkpoints


def validate_camera_data(camera: str, base_data_dir: str = "data") -> bool:
    """
    Validate that camera data directory exists and has checkpoints.
    
    Args:
        camera: Camera identifier
        base_data_dir: Base data directory
        
    Returns:
        True if valid, False otherwise
    """
    # Handle camera path structure: data/APN/APN_K024/30/test.json  
    camera_parts = camera.split('_')
    if len(camera_parts) >= 2:
        camera_group = camera_parts[0]  # e.g., 'APN'
        camera_dir = os.path.join(base_data_dir, camera_group, camera)
        test_file = os.path.join(camera_dir, "30", "test.json")
    else:
        camera_dir = os.path.join(base_data_dir, camera)
        test_file = os.path.join(camera_dir, "30", "test.json")
    
    if not os.path.exists(camera_dir):
        logger.error(f"Camera directory does not exist: {camera_dir}")
        return False
    
    if not os.path.exists(test_file):
        logger.error(f"Test data file does not exist: {test_file}")
        return False
    
    checkpoints = get_checkpoint_directories(camera, base_data_dir)
    if not checkpoints:
        logger.error(f"No checkpoints found for camera: {camera}")
        return False
    
    # logger.info(f"Camera data validation passed for {camera}")
    return True


def ensure_directory_exists(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path of the directory
    """
    abs_dir = os.path.abspath(directory)
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


def get_results_file_path(log_dir: str, filename: str = "results.json") -> str:
    """
    Get path for results file in log directory.
    
    Args:
        log_dir: Log directory path
        filename: Results filename
        
    Returns:
        Full path to results file
    """
    return os.path.join(log_dir, filename)


def get_log_file_path(log_dir: str, filename: str = "log.txt") -> str:
    """
    Get path for log file in log directory.
    
    Args:
        log_dir: Log directory path
        filename: Log filename
        
    Returns:
        Full path to log file
    """
    return os.path.join(log_dir, filename)


def get_predictions_file_path(log_dir: str, checkpoint: str) -> str:
    """
    Get path for predictions file for a specific checkpoint.
    
    Args:
        log_dir: Log directory path
        checkpoint: Checkpoint name
        
    Returns:
        Full path to predictions file
    """
    return os.path.join(log_dir, f"predictions_{checkpoint}.json")


def get_relative_path(full_path: str, base_path: str = None) -> str:
    """
    Get relative path from full path.
    
    Args:
        full_path: Full absolute path
        base_path: Base path to make relative to (default: current working directory)
        
    Returns:
        Relative path
    """
    if base_path is None:
        base_path = os.getcwd()
    
    try:
        return os.path.relpath(full_path, base_path)
    except ValueError:
        # Return full path if relative path can't be determined
        return full_path


def get_relative_path(full_path: str, base_path: str = None) -> str:
    """
    Get relative path from full path.
    
    Args:
        full_path: Full absolute path
        base_path: Base path to make relative to (default: current working directory)
        
    Returns:
        Relative path
    """
    if base_path is None:
        base_path = os.getcwd()
    
    try:
        return os.path.relpath(full_path, base_path)
    except ValueError:
        # Return full path if relative path can't be determined
        return full_path
