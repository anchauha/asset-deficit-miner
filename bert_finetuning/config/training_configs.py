"""
Default training configurations for BERT fine-tuning.

This file contains pre-defined configurations for different training scenarios.
You can modify these or create new configurations as needed.
"""

# Base configuration for BERT fine-tuning
DEFAULT_CONFIG = {
    "model_name": "bert-base-uncased",
    "num_labels": 5,  # O, B-ASSET, I-ASSET, B-DEFICIT, I-DEFICIT
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "device": "auto",
    "save_best_only": True,
    "patience": 3,
    "eval_strategy": "epoch",
    "eval_steps": 500,
    "logging_steps": 100
}

# Quick training configuration for testing
QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    "num_epochs": 3,
    "batch_size": 8,
    "patience": 2
}

# High-performance configuration for production
PRODUCTION_CONFIG = {
    **DEFAULT_CONFIG,
    "model_name": "bert-large-uncased",
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-5,
    "patience": 5,
    "warmup_ratio": 0.15
}

# GPU-optimized configuration
GPU_OPTIMIZED_CONFIG = {
    **DEFAULT_CONFIG,
    "batch_size": 32,
    "num_epochs": 15,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "logging_steps": 50
}

# CPU configuration for systems without GPU
CPU_CONFIG = {
    **DEFAULT_CONFIG,
    "batch_size": 4,
    "num_epochs": 5,
    "device": "cpu",
    "eval_steps": 100,
    "logging_steps": 25
}

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "context_window": 200,
    "val_split": 0.2,
    "shuffle_data": True,
    "random_seed": 42
}

# Inference configuration
INFERENCE_CONFIG = {
    "confidence_threshold": 0.5,
    "device": "auto",
    "batch_size": 16,
    "max_length": 512,
    "sliding_window_size": 400,
    "sliding_window_overlap": 50
}

# Web service configuration
WEB_SERVICE_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": False,
    "threaded": True,
    "device": "auto",
    "confidence_threshold": 0.5
}

# Configuration mapping for easy access
CONFIGS = {
    "default": DEFAULT_CONFIG,
    "quick": QUICK_CONFIG,
    "production": PRODUCTION_CONFIG,
    "gpu": GPU_OPTIMIZED_CONFIG,
    "cpu": CPU_CONFIG
}

def get_config(config_name: str = "default"):
    """
    Get a pre-defined configuration.
    
    Args:
        config_name: Name of the configuration ('default', 'quick', 'production', 'gpu', 'cpu')
        
    Returns:
        Configuration dictionary
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")
    
    return CONFIGS[config_name].copy()

def create_custom_config(**kwargs):
    """
    Create a custom configuration by overriding default values.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        Custom configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config