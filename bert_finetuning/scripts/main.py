#!/usr/bin/env python3
"""
Main Pipeline for BERT Fine-tuning on Asset/Deficit Classification.

This script provides a complete end-to-end pipeline for:
1. Processing JSON data with asset/deficit annotations
2. Training a BERT model for token classification
3. Running inference on new documents
4. Generating HTML visualizations

Usage Examples:
    # Process data only
    python scripts/main.py process --input data/input/data.json --output data/processed/

    # Train model
    python scripts/main.py train --input data/input/data.json --model-dir models/ --epochs 10

    # Run inference
    python scripts/main.py predict --model models/best_model --text "Your text here" --output outputs/result.html

    # Complete pipeline
    python scripts/main.py pipeline --input data/input/data.json --output outputs/results/
"""

import argparse
import json
import os
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path

try:
    from dataclasses import asdict
except ImportError:
    # Fallback for older Python versions
    def asdict(obj):
        return obj.__dict__

# Add project root to Python path for standalone operation
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.data_processor import AssetDeficitDataProcessor
from config.trainer import AssetDeficitTrainer, TrainingConfig
from config.inference import TextVisualizer  # Keep for other uses if needed
from config.prediction import StandaloneBERTPredictor  # New standalone predictor
from config.helpers import setup_logging, save_json, load_json, create_directory_if_not_exists

def setup_pipeline_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Set up logging for the pipeline."""
    log_file = os.path.join(output_dir, "pipeline.log")
    return setup_logging(log_level, log_file)

def process_data_command(args) -> Dict[str, Any]:
    """Process raw JSON data into training examples."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing data from {args.input}")
    
    # Create output directory
    create_directory_if_not_exists(args.output)
    
    # Initialize processor
    processor = AssetDeficitDataProcessor(
        model_name=args.model_name,
        max_length=args.max_length,
        context_window=args.context_window
    )
    
    # Process the data
    examples = processor.process_json_file(args.input)
    
    # Save processed examples with processing configuration
    output_file = os.path.join(args.output, "processed_examples.json")
    processed_data = {
        "processing_config": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "context_window": args.context_window
        },
        "examples": [
            {
                "text": ex.text,
                "tokens": ex.tokens,
                "labels": ex.labels,
                "metadata": ex.metadata
            }
            for ex in examples
        ]
    }
    
    save_json(processed_data, output_file)
    
    # Save processing statistics
    stats = {
        "total_examples": len(examples),
        "total_tokens": sum(len(ex.tokens) for ex in examples),
        "label_distribution": processor.get_label_statistics(examples),
        "processing_config": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "context_window": args.context_window
        }
    }
    
    stats_file = os.path.join(args.output, "processing_stats.json")
    save_json(stats, stats_file)
    
    logger.info(f"Processed {len(examples)} examples")
    logger.info(f"Results saved to {args.output}")
    
    return stats

def train_model_command(args) -> Dict[str, Any]:
    """Train a BERT model for asset/deficit classification."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training model with data from {args.input}")
    
    # Create model directory
    create_directory_if_not_exists(args.model_dir)
    
    # Initialize processor
    processor = AssetDeficitDataProcessor(
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Process data if not already processed
    if args.input.endswith('.json') and 'processed_examples.json' not in args.input:
        # Raw JSON file - max_length is required
        if not args.max_length:
            raise ValueError("--max-length is required when training with raw JSON files")
        examples = processor.process_json_file(args.input)
        effective_max_length = args.max_length
    else:
        # Load pre-processed examples and extract saved max_length
        processed_data = load_json(args.input)
        
        # Handle both old format (list) and new format (dict with config)
        if isinstance(processed_data, list):
            # Old format - fallback to args.max_length or default
            examples_data = processed_data
            if not args.max_length:
                effective_max_length = 256  # fallback default
                logger.warning(f"Using old processed format. Defaulting max_length to {effective_max_length}")
            else:
                effective_max_length = args.max_length
                logger.info(f"Using user-specified max_length: {effective_max_length}")
        else:
            # New format with saved config
            examples_data = processed_data['examples']
            saved_config = processed_data['processing_config']
            saved_max_length = saved_config['max_length']
            
            if args.max_length:
                # User wants to override the saved max_length
                effective_max_length = args.max_length
                logger.info(f"Overriding saved max_length ({saved_max_length}) with user-specified: {effective_max_length}")
            else:
                # Use saved max_length
                effective_max_length = saved_max_length
                logger.info(f"Using saved processing max_length: {effective_max_length}")
            
            # Update processor with saved config to ensure consistency
            processor = AssetDeficitDataProcessor(
                model_name=saved_config['model_name'],
                max_length=effective_max_length
            )
        
        examples = processor.load_processed_examples(examples_data)
    
    # Split data
    train_examples, val_examples = processor.split_data(examples, val_ratio=args.val_split)
    
    # Create data loaders using the effective max_length
    train_loader = processor.create_data_loader(
        train_examples, 
        batch_size=args.batch_size, 
        shuffle=True,
        max_length=effective_max_length
    )
    val_loader = processor.create_data_loader(
        val_examples, 
        batch_size=args.batch_size, 
        shuffle=False,
        max_length=effective_max_length
    )
    
    # Initialize training configuration
    config = TrainingConfig(
        model_name=args.model_name,
        num_labels=len(processor.label_to_id),
        max_length=effective_max_length,  # Use effective max_length for model config
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_dir=args.model_dir,
        device=args.device
    )
    
    # Initialize trainer
    trainer = AssetDeficitTrainer(config)
    
    # Train the model
    training_history = trainer.train(train_loader, val_loader)
    
    # Save comprehensive training results in the best model directory
    best_model_dir = os.path.join(args.model_dir, "best_model")
    results = {
        "experiment_info": {
            "input_file": args.input,
            "model_directory": args.model_dir,
            "training_completed": True,
            "best_f1_score": trainer.best_f1
        },
        "training_config": config.__dict__,
        "training_history": training_history,
        "data_stats": {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "total_examples": len(examples),
            "validation_split": len(val_examples) / len(examples)
        },
        "model_performance": {
            "best_f1": trainer.best_f1,
            "total_epochs": len(training_history.get('train_loss', [])),
            "final_train_loss": training_history.get('train_loss', [])[-1] if training_history.get('train_loss') else None,
            "final_val_loss": training_history.get('val_loss', [])[-1] if training_history.get('val_loss') else None
        }
    }
    
    # Save main results with the best model (most important for inference)
    results_file = os.path.join(best_model_dir, "experiment_results.json")
    save_json(results, results_file)
    
    # Also save a summary in the root model directory for easy access
    summary_file = os.path.join(args.model_dir, "experiment_summary.json")
    summary = {
        "best_f1_score": trainer.best_f1,
        "total_epochs": len(training_history.get('train_loss', [])),
        "model_locations": {
            "best_model": best_model_dir,
            "final_model": os.path.join(args.model_dir, "final_model")
        },
        "input_data": args.input,
        "completed_successfully": True
    }
    save_json(summary, summary_file)
    
    logger.info(f"Training completed. Model saved to {args.model_dir}")
    
    return results

def predict_command(args) -> Dict[str, Any]:
    """Run inference on text and generate visualization."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running inference with model from {args.model}")
    
    # Initialize standalone predictor (doesn't interfere with existing modules)
    device = getattr(args, 'device', 'auto')
    predictor = StandaloneBERTPredictor(args.model, device=device)
    
    # Get text input
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Must provide either --text or --text-file")
    
    # Run prediction
    confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
    spans = predictor.predict_text(text, confidence_threshold=confidence_threshold)
    
    # Generate HTML visualization if output file provided
    if args.output and args.output.endswith('.html'):
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            create_directory_if_not_exists(output_dir)
        
        # Use the standalone predictor's HTML generation
        title = args.title or "BERT Asset/Deficit Analysis"
        predictor.create_html_report(text, spans, args.output, title)
        
        logger.info(f"HTML visualization saved to {args.output}")
    
    # Save spans data
    if args.output:
        base_name = args.output.replace('.html', '')
        spans_file = f"{base_name}_spans.json"
        spans_data = [asdict(span) for span in spans]  # Use asdict for dataclasses
        save_json(spans_data, spans_file)
        logger.info(f"Spans data saved to {spans_file}")
    
    return {"spans": [asdict(span) for span in spans]}

def pipeline_command(args) -> Dict[str, Any]:
    """Run the complete pipeline: process, train, and predict."""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete pipeline")
    
    # Create output directory structure
    create_directory_if_not_exists(args.output)
    
    # Step 1: Process data
    logger.info("Step 1: Processing data")
    processed_dir = os.path.join(args.output, "processed_data")
    process_args = argparse.Namespace(
        input=args.input,
        output=processed_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        context_window=args.context_window
    )
    processing_stats = process_data_command(process_args)
    
    # Step 2: Train model
    logger.info("Step 2: Training model")
    model_dir = os.path.join(args.output, "trained_model")
    train_args = argparse.Namespace(
        input=os.path.join(processed_dir, "processed_examples.json"),
        model_dir=model_dir,
        model_name=args.model_name,
        max_length=None,  # Will use saved max_length from processed data
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device
    )
    training_results = train_model_command(train_args)
    
    # Step 3: Run inference on sample text if provided
    if args.sample_text or args.sample_text_file:
        logger.info("Step 3: Running sample inference")
        predict_args = argparse.Namespace(
            model=os.path.join(model_dir, "best_model"),
            text=args.sample_text,
            text_file=args.sample_text_file,
            output=os.path.join(args.output, "analysis_results.html"),
            title="Pipeline Analysis Results"
        )
        prediction_results = predict_command(predict_args)
    else:
        prediction_results = None
    
    # Save complete pipeline configuration and results
    pipeline_results = {
        "pipeline_config": {
            "input_file": args.input,
            "output_directory": args.output,
            "model_name": args.model_name,
            "training_epochs": args.epochs,
            "batch_size": args.batch_size
        },
        "processing_stats": processing_stats,
        "training_results": training_results,
        "prediction_results": prediction_results
    }
    
    pipeline_file = os.path.join(args.output, "pipeline_results.json")
    save_json(pipeline_results, pipeline_file)
    
    logger.info(f"Pipeline completed successfully. Results in {args.output}")
    
    return pipeline_results

def main():
    """Main entry point for the BERT fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description="BERT Fine-tuning Pipeline for Asset/Deficit Classification")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments for all commands
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    # Common arguments for commands that use models/tokenizers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--model-name', default='bert-base-uncased', help='BERT model name')
    
    # Common arguments for commands that need GPU/device selection
    device_parser = argparse.ArgumentParser(add_help=False)
    device_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    
    # Process command (CPU-only, no device parameter needed)
    process_parser = subparsers.add_parser('process', parents=[base_parser, model_parser], help='Process raw data')
    process_parser.add_argument('--input', required=True, help='Input JSON file')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length for tokenization')
    process_parser.add_argument('--context-window', type=int, default=200, help='Context window around spans')
    
    # Train command (needs device for GPU training)
    train_parser = subparsers.add_parser('train', parents=[base_parser, model_parser, device_parser], help='Train BERT model')
    train_parser.add_argument('--input', required=True, help='Input data (raw JSON file or processed_examples.json)')
    train_parser.add_argument('--model-dir', required=True, help='Model output directory')
    train_parser.add_argument('--max-length', type=int, help='Maximum sequence length (required for raw JSON, auto-detected for processed data, can override saved value)')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    train_parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    train_parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    # Predict command (needs device for model inference)
    predict_parser = subparsers.add_parser('predict', parents=[base_parser, device_parser], help='Run inference')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--text', help='Text to analyze')
    predict_parser.add_argument('--text-file', help='File containing text to analyze')
    predict_parser.add_argument('--output', help='Output file (HTML for visualization)')
    predict_parser.add_argument('--title', help='Title for HTML visualization')
    
    # Pipeline command (needs device for training phase)
    pipeline_parser = subparsers.add_parser('pipeline', parents=[base_parser, model_parser, device_parser], help='Run complete pipeline')
    pipeline_parser.add_argument('--input', required=True, help='Input JSON file')
    pipeline_parser.add_argument('--output', required=True, help='Output directory')
    pipeline_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length for tokenization')
    pipeline_parser.add_argument('--context-window', type=int, default=200, help='Context window around spans')
    pipeline_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    pipeline_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    pipeline_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    pipeline_parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    pipeline_parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    pipeline_parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    pipeline_parser.add_argument('--sample-text', help='Sample text for inference demonstration')
    pipeline_parser.add_argument('--sample-text-file', help='File with sample text for inference')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    if hasattr(args, 'output') and args.output:
        # For predict command, output is a file, not a directory
        if args.command == 'predict':
            # For predict, use the directory containing the output file for logging
            if args.output.endswith('.html'):
                output_dir = os.path.dirname(args.output) or '.'
                # Ensure the output directory exists for logging
                if output_dir != '.':
                    create_directory_if_not_exists(output_dir)
                logger = setup_pipeline_logging(output_dir, args.log_level)
            else:
                logger = setup_logging(args.log_level)
        else:
            # For other commands, output is typically a directory
            create_directory_if_not_exists(args.output)
            logger = setup_pipeline_logging(args.output, args.log_level)
    else:
        logger = setup_logging(args.log_level)
    
    try:
        # Execute command
        if args.command == 'process':
            process_data_command(args)
        elif args.command == 'train':
            train_model_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'pipeline':
            pipeline_command(args)
        
        logger.info(f"Command '{args.command}' completed successfully")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {str(e)}")
        raise

if __name__ == "__main__":
    main()