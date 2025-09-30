"""
BERT Fine-tuning Trainer for Asset/Deficit Classification.

This module provides a comprehensive trainer for fine-tuning BERT models on 
token classification tasks, specifically for identifying asset and deficit 
spans in text documents.

Features:
- Handles class imbalance with weighted loss
- Comprehensive evaluation metrics
- Model checkpointing and saving
- Learning rate scheduling
- Gradient clipping for stability

Usage:
    config = TrainingConfig(num_epochs=10, batch_size=16)
    trainer = AssetDeficitTrainer(config)
    trainer.train(train_loader, val_loader)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, f1_score
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    model_name: str = 'bert-base-uncased'
    num_labels: int = 5  # O, B-ASSET, I-ASSET, B-DEFICIT, I-DEFICIT
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_dir: str = './models'
    device: str = 'auto'
    save_best_only: bool = True
    patience: int = 3
    eval_strategy: str = 'epoch'  # 'epoch' or 'steps'
    eval_steps: int = 500
    logging_steps: int = 100
    
    def __post_init__(self):
        """Set device automatically if needed."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

class AssetDeficitTrainer:
    """
    Trainer for BERT fine-tuning on asset/deficit classification.
    
    Features:
    - Automatic class weight calculation for imbalanced data
    - Learning rate scheduling with warmup
    - Model checkpointing and early stopping
    - Comprehensive evaluation metrics
    - Progress tracking and logging
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Training state
        self.global_step = 0
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model: {config.model_name}, Labels: {config.num_labels}")
    
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        
        # Calculate class weights for imbalanced data
        class_weights = self._calculate_class_weights(train_loader)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, criterion)
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader, criterion)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_f1'].append(val_metrics['weighted avg']['f1-score'])
            self.training_history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Log progress
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val F1: {val_metrics['weighted avg']['f1-score']:.4f}")
            
            # Save checkpoint and check early stopping
            current_f1 = val_metrics['weighted avg']['f1-score']
            if self._should_save_model(current_f1):
                self._save_model('best_model')
                self.best_f1 = current_f1
                self.patience_counter = 0
                logger.info(f"New best model saved! F1: {current_f1:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self._save_model('final_model')
        
        # Save training history
        self._save_training_history()
        
        logger.info("Training completed!")
        return self.training_history
    
    def _train_epoch(self, train_loader, optimizer, scheduler, criterion) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Calculate loss
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update progress
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                logger.debug(f"Step {self.global_step}, Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader, criterion) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Flatten and filter out ignored labels
                flat_predictions = predictions.view(-1).cpu().numpy()
                flat_labels = batch['labels'].view(-1).cpu().numpy()
                
                # Remove ignored labels (-100)
                mask = flat_labels != -100
                flat_predictions = flat_predictions[mask]
                flat_labels = flat_labels[mask]
                
                all_predictions.extend(flat_predictions)
                all_labels.extend(flat_labels)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        # Convert to label names for classification report
        label_names = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
        
        # Create classification report
        report = classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        
        return avg_loss, report
    
    def _calculate_class_weights(self, train_loader) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        label_counts = {}
        total_labels = 0
        
        for batch in train_loader:
            labels = batch['labels'].view(-1)
            # Count non-ignored labels
            valid_labels = labels[labels != -100]
            
            for label_id in valid_labels:
                label_id = label_id.item()
                label_counts[label_id] = label_counts.get(label_id, 0) + 1
                total_labels += 1
        
        # Calculate weights (inverse frequency)
        weights = torch.ones(self.config.num_labels)
        for label_id, count in label_counts.items():
            weights[label_id] = total_labels / (len(label_counts) * count)
        
        # Clamp weights to prevent extreme values that can cause training instability
        weights = torch.clamp(weights, 0.1, 10.0)
        
        weights = weights.to(self.device)
        
        logger.info(f"Class weights (clamped): {weights}")
        # Log detailed weight information
        label_names = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
        for i, (label, weight) in enumerate(zip(label_names, weights)):
            count = label_counts.get(i, 0)
            logger.info(f"  {label}: {weight:.4f} (count: {count})")
        
        return weights
    
    def _should_save_model(self, current_f1: float) -> bool:
        """Check if current model should be saved."""
        if self.config.save_best_only:
            return current_f1 > self.best_f1
        return True
    
    def _save_model(self, model_name: str):
        """Save the model and tokenizer with training statistics."""
        save_path = os.path.join(self.config.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training configuration separately (don't overwrite model config.json)
        training_config_dict = {
            'model_name': self.config.model_name,
            'num_labels': self.config.num_labels,
            'max_length': self.config.max_length,
            'label_names': ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT'],
            'training_params': self.config.__dict__
        }
        
        with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
            json.dump(training_config_dict, f, indent=2)
        
        # Save training statistics with this specific model
        self._save_training_statistics(save_path, model_name)
        
        logger.debug(f"Model and statistics saved to {save_path}")
    
    def _save_training_statistics(self, save_path: str, model_name: str):
        """Save training statistics to the model directory."""
        # Save current training history
        history_file = os.path.join(save_path, 'training_history.json')
        history_with_config = {
            'model_type': model_name,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_f1': self.best_f1,
            'total_steps': self.global_step,
            'current_epoch': len(self.training_history['train_loss']),
            'patience_counter': self.patience_counter
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_with_config, f, indent=2)
        
        # Save detailed training metrics
        metrics_file = os.path.join(save_path, 'training_metrics.json')
        detailed_metrics = {
            'final_metrics': {
                'best_f1_score': self.best_f1,
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
                'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
                'final_val_f1': self.training_history['val_f1'][-1] if self.training_history['val_f1'] else None,
            },
            'training_progress': self.training_history,
            'model_info': {
                'model_name': self.config.model_name,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
            
        logger.info(f"Training statistics saved to {save_path}")
    
    def _save_training_history(self):
        """Save overall training history summary to main model directory."""
        history_file = os.path.join(self.config.save_dir, 'training_summary.json')
        
        # Create a comprehensive summary
        summary = {
            'experiment_summary': {
                'total_epochs_completed': len(self.training_history['train_loss']),
                'best_f1_achieved': self.best_f1,
                'early_stopping_triggered': self.patience_counter >= self.config.patience,
                'final_patience_counter': self.patience_counter
            },
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'model_locations': {
                'best_model': os.path.join(self.config.save_dir, 'best_model'),
                'final_model': os.path.join(self.config.save_dir, 'final_model')
            }
        }
        
        with open(history_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {history_file}")
        logger.info(f"Detailed statistics saved with each model in subdirectories")

def create_trainer_from_config(config_dict: Dict) -> AssetDeficitTrainer:
    """Create trainer from configuration dictionary."""
    config = TrainingConfig(**config_dict)
    return AssetDeficitTrainer(config)