"""
Utility functions for loading configuration and setting random seeds.
"""
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but non-deterministic
        torch.backends.cudnn.benchmark = True


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get computing device (CUDA/CPU).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device
    """
    if config['gpu']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for the project.
    
    Args:
        config: Configuration dictionary
    """
    dirs = [
        config['dataset']['cache_dir'],
        config['checkpoint']['save_dir'],
        config['logging']['log_dir'],
        './results',
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Created project directories")
