"""
Logging utilities for ICICLE-Benchmark V2
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import sys


def setup_logging(output_dir: str, debug: bool = False, experiment_name: str = "experiment") -> logging.Logger:
    """
    Setup logging for the pipeline.
    
    Args:
        output_dir: Directory to save logs
        debug: Whether to enable debug level logging
        experiment_name: Name of the experiment for log files
    
    Returns:
        Configured logger
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging level
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger
