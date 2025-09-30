"""
Utility functions for the BERT fine-tuning pipeline.
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)

def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"experiment_{timestamp}"
    exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_input_file(file_path: str, required_keys: List[str] = None) -> bool:
    """
    Validate that input JSON file has required structure.
    
    Args:
        file_path: Path to JSON file
        required_keys: List of required keys in the JSON
        
    Returns:
        True if file is valid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if required_keys:
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Required key '{key}' not found in input file")
        
        return True
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }