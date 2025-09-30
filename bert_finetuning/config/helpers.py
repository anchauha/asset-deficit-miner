"""
Utility functions for the BERT fine-tuning pipeline.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    # Ensure directory exists (only if there's actually a directory path)
    dir_path = os.path.dirname(file_path)
    if dir_path:  # Only create directory if path is not empty
        os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory_if_not_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    os.makedirs(directory, exist_ok=True)

def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Parameter count dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def validate_file_exists(file_path: str) -> bool:
    """
    Check if file exists.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists
    """
    return os.path.isfile(file_path)

def validate_directory_exists(directory: str) -> bool:
    """
    Check if directory exists.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if directory exists
    """
    return os.path.isdir(directory)

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path) if validate_file_exists(file_path) else 0

def create_experiment_directory(base_dir: str, experiment_name: str = None) -> str:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to created experiment directory
    """
    timestamp = get_timestamp()
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, dir_name)
    create_directory_if_not_exists(experiment_dir)
    
    return experiment_dir

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (JSON or Python)
        
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.json'):
        return load_json(config_path)
    elif config_path.endswith('.py'):
        # Load Python config file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract configuration
        config = {}
        for attr in dir(config_module):
            if not attr.startswith('_'):
                config[attr] = getattr(config_module, attr)
        
        return config
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def calculate_metrics(predictions: List[int], labels: List[int], label_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        label_names: Optional label names
        
    Returns:
        Metrics dictionary
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Detailed report if label names provided
    if label_names:
        report = classification_report(labels, predictions, target_names=label_names, output_dict=True)
        metrics['detailed_report'] = report
    
    return metrics

def print_training_progress(epoch: int, total_epochs: int, step: int, total_steps: int, 
                          loss: float, metrics: Dict[str, float] = None) -> None:
    """
    Print training progress.
    
    Args:
        epoch: Current epoch
        total_epochs: Total epochs
        step: Current step
        total_steps: Total steps
        loss: Current loss
        metrics: Optional metrics dictionary
    """
    progress_str = f"Epoch {epoch}/{total_epochs} | Step {step}/{total_steps} | Loss: {loss:.4f}"
    
    if metrics:
        for key, value in metrics.items():
            progress_str += f" | {key}: {value:.4f}"
    
    print(progress_str)

def save_model_info(model, tokenizer, save_dir: str, config: Dict[str, Any] = None) -> None:
    """
    Save model information and configuration.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        save_dir: Directory to save to
        config: Optional configuration dictionary
    """
    create_directory_if_not_exists(save_dir)
    
    # Model info
    param_counts = count_parameters(model)
    
    model_info = {
        'model_type': type(model).__name__,
        'parameters': param_counts,
        'model_size_mb': param_counts['total'] * 4 / (1024 * 1024),  # Rough estimate
        'tokenizer_type': type(tokenizer).__name__,
        'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else None,
        'created_at': datetime.now().isoformat()
    }
    
    if config:
        model_info['training_config'] = config
    
    # Save info
    info_path = os.path.join(save_dir, 'model_info.json')
    save_json(model_info, info_path)
    
    print(f"Model info saved to {info_path}")
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    
    Returns:
        GPU memory info dictionary
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            memory_info = {}
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)
                
                memory_info[f'gpu_{i}'] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'allocated_memory': allocated,
                    'cached_memory': cached,
                    'free_memory': props.total_memory - allocated
                }
            
            return memory_info
        else:
            return {'message': 'CUDA not available'}
    except ImportError:
        return {'message': 'PyTorch not available'}

def format_time(seconds: float) -> str:
    """
    Format time duration.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_run_summary(config: Dict[str, Any], metrics: Dict[str, Any], 
                      duration: float, save_dir: str) -> None:
    """
    Create a summary of the training run.
    
    Args:
        config: Training configuration
        metrics: Final metrics
        duration: Training duration in seconds
        save_dir: Directory to save summary
    """
    summary = {
        'run_info': {
            'duration': format_time(duration),
            'duration_seconds': duration,
            'completed_at': datetime.now().isoformat()
        },
        'config': config,
        'final_metrics': metrics,
        'system_info': {
            'gpu_memory': get_gpu_memory_info()
        }
    }
    
    summary_path = os.path.join(save_dir, 'run_summary.json')
    save_json(summary, summary_path)
    
    print(f"Run summary saved to {summary_path}")
    print(f"Training completed in {format_time(duration)}")
    
    if 'f1' in metrics:
        print(f"Final F1 score: {metrics['f1']:.4f}")
    if 'accuracy' in metrics:
        print(f"Final accuracy: {metrics['accuracy']:.4f}")