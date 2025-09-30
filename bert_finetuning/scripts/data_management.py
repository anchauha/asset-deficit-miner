#!/usr/bin/env python3
"""
Data Management Script for BERT Fine-tuning Pipeline.

This script demonstrates how to work with the data processing pipeline,
including loading, processing, and validating training data.

Usage:
    python scripts/data_management.py --input data/input/sample_data.json --output data/processed/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.data_processor import AssetDeficitDataProcessor
from config.helpers import setup_logging, save_json, create_directory_if_not_exists

def main():
    """Main function for data management operations."""
    parser = argparse.ArgumentParser(description="Data Management for BERT Fine-tuning")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw data into training examples')
    process_parser.add_argument('--input', required=True, help='Input JSON file')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--model-name', default='bert-base-uncased', help='BERT model name')
    process_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    process_parser.add_argument('--context-window', type=int, default=200, help='Context window around spans')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data format')
    validate_parser.add_argument('--input', required=True, help='Input JSON file to validate')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate data statistics')
    stats_parser.add_argument('--input', required=True, help='Input JSON file')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between data formats')
    convert_parser.add_argument('--input', required=True, help='Input file')
    convert_parser.add_argument('--output', required=True, help='Output file')
    convert_parser.add_argument('--from-format', choices=['json', 'conll'], help='Input format')
    convert_parser.add_argument('--to-format', choices=['json', 'conll'], help='Output format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    logger = setup_logging()
    
    try:
        if args.command == 'process':
            process_data(args, logger)
        elif args.command == 'validate':
            validate_data(args, logger)
        elif args.command == 'stats':
            generate_stats(args, logger)
        elif args.command == 'convert':
            convert_data(args, logger)
            
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {str(e)}")
        raise

def process_data(args, logger):
    """Process raw data into training examples."""
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
    
    # Save processed examples
    output_file = os.path.join(args.output, "processed_examples.json")
    examples_data = [
        {
            "text": ex.text,
            "tokens": ex.tokens,
            "labels": ex.labels,
            "token_positions": ex.token_positions,
            "metadata": ex.metadata
        }
        for ex in examples
    ]
    
    save_json(examples_data, output_file)
    
    # Save processing statistics
    stats = {
        "total_examples": len(examples),
        "total_tokens": sum(len(ex.tokens) for ex in examples),
        "label_distribution": processor.get_label_statistics(examples),
        "processing_config": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "context_window": args.context_window
        },
        "example_types": {
            "span_focused": len([ex for ex in examples if ex.metadata.get('example_type') == 'span_focused']),
            "full_document": len([ex for ex in examples if ex.metadata.get('example_type') == 'full_document']),
            "no_spans": len([ex for ex in examples if ex.metadata.get('example_type') == 'no_spans'])
        }
    }
    
    stats_file = os.path.join(args.output, "processing_stats.json")
    save_json(stats, stats_file)
    
    logger.info(f"Processed {len(examples)} examples")
    logger.info(f"Label distribution: {stats['label_distribution']}")
    logger.info(f"Example types: {stats['example_types']}")
    logger.info(f"Results saved to {args.output}")

def validate_data(args, logger):
    """Validate data format and content."""
    logger.info(f"Validating data from {args.input}")
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return False
    
    # Validate structure
    errors = []
    warnings = []
    
    if 'data' not in data:
        errors.append("Missing 'data' field at root level")
        return False
    
    documents = data['data']
    if not isinstance(documents, list):
        errors.append("'data' field must be a list")
        return False
    
    for i, doc in enumerate(documents):
        doc_errors = validate_document(doc, i)
        errors.extend(doc_errors)
    
    # Report results
    if errors:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    if warnings:
        logger.warning("Validation warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    logger.info(f"Validation passed! Found {len(documents)} valid documents")
    return True

def validate_document(doc: Dict[str, Any], index: int) -> List[str]:
    """Validate a single document."""
    errors = []
    
    required_fields = ['Sr', 'source', 'extracted_content']
    for field in required_fields:
        if field not in doc:
            errors.append(f"Document {index}: Missing required field '{field}'")
    
    if 'extracted_content' in doc and not doc['extracted_content'].strip():
        errors.append(f"Document {index}: Empty extracted_content")
    
    # Check linguistic classification
    if 'linguistic_classification' in doc:
        classification = doc['linguistic_classification']
        if 'analysis' in classification:
            for j, analysis in enumerate(classification['analysis']):
                if 'text_span' not in analysis:
                    errors.append(f"Document {index}, analysis {j}: Missing 'text_span'")
                if 'type' not in analysis:
                    errors.append(f"Document {index}, analysis {j}: Missing 'type'")
                elif analysis['type'] not in ['Asset', 'Deficit']:
                    errors.append(f"Document {index}, analysis {j}: Invalid type '{analysis['type']}'")
    
    return errors

def generate_stats(args, logger):
    """Generate statistics about the data."""
    logger.info(f"Generating statistics for {args.input}")
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = data.get('data', [])
    
    # Basic statistics
    stats = {
        "dataset_info": {
            "total_documents": len(documents),
            "total_characters": sum(len(doc.get('extracted_content', '')) for doc in documents),
            "avg_document_length": sum(len(doc.get('extracted_content', '')) for doc in documents) / len(documents) if documents else 0
        },
        "span_statistics": {
            "total_spans": 0,
            "asset_spans": 0,
            "deficit_spans": 0,
            "span_lengths": []
        },
        "documents_with_spans": 0,
        "documents_without_spans": 0
    }
    
    # Analyze spans
    for doc in documents:
        has_spans = False
        classification = doc.get('linguistic_classification', {})
        analysis = classification.get('analysis', [])
        
        for span in analysis:
            stats["span_statistics"]["total_spans"] += 1
            span_type = span.get('type', '')
            span_text = span.get('text_span', '')
            
            if span_type == 'Asset':
                stats["span_statistics"]["asset_spans"] += 1
            elif span_type == 'Deficit':
                stats["span_statistics"]["deficit_spans"] += 1
            
            stats["span_statistics"]["span_lengths"].append(len(span_text))
            has_spans = True
        
        if has_spans:
            stats["documents_with_spans"] += 1
        else:
            stats["documents_without_spans"] += 1
    
    # Calculate span length statistics
    if stats["span_statistics"]["span_lengths"]:
        import statistics
        lengths = stats["span_statistics"]["span_lengths"]
        stats["span_statistics"]["avg_span_length"] = statistics.mean(lengths)
        stats["span_statistics"]["median_span_length"] = statistics.median(lengths)
        stats["span_statistics"]["min_span_length"] = min(lengths)
        stats["span_statistics"]["max_span_length"] = max(lengths)
    
    # Print statistics
    logger.info("Dataset Statistics:")
    logger.info(f"  Total documents: {stats['dataset_info']['total_documents']}")
    logger.info(f"  Average document length: {stats['dataset_info']['avg_document_length']:.1f} characters")
    logger.info(f"  Documents with spans: {stats['documents_with_spans']}")
    logger.info(f"  Documents without spans: {stats['documents_without_spans']}")
    logger.info("")
    logger.info("Span Statistics:")
    logger.info(f"  Total spans: {stats['span_statistics']['total_spans']}")
    logger.info(f"  Asset spans: {stats['span_statistics']['asset_spans']}")
    logger.info(f"  Deficit spans: {stats['span_statistics']['deficit_spans']}")
    
    if stats["span_statistics"]["span_lengths"]:
        logger.info(f"  Average span length: {stats['span_statistics']['avg_span_length']:.1f} characters")
        logger.info(f"  Span length range: {stats['span_statistics']['min_span_length']}-{stats['span_statistics']['max_span_length']} characters")
    
    # Save detailed statistics
    output_dir = os.path.dirname(args.input)
    stats_file = os.path.join(output_dir, "dataset_statistics.json")
    save_json(stats, stats_file)
    logger.info(f"Detailed statistics saved to {stats_file}")

def convert_data(args, logger):
    """Convert between different data formats."""
    logger.info(f"Converting {args.input} from {args.from_format} to {args.to_format}")
    
    if args.from_format == args.to_format:
        logger.warning("Input and output formats are the same")
        return
    
    # Implementation for format conversion would go here
    # This is a placeholder for future functionality
    logger.warning("Format conversion not yet implemented")

if __name__ == "__main__":
    main()