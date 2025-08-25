"""
Training module for Camera Trap Framework V2.
Contains Oracle and Accumulative training implementations.
"""

from .common import *
from .oracle import train_model_oracle
from .accumulative import train_model_accumulative

__all__ = ['train_model_oracle', 'train_model_accumulative']
