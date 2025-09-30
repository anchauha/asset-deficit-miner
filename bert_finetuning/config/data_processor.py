"""
Enhanced data processor for BERT fine-tuning that solves the 512 token limit problem.

This module processes JSON data containing documents with asset/deficit span annotations
and creates training examples that can handle documents longer than BERT's 512 token limit.

The key innovation is a span-focused approach:
- Instead of truncating long documents (losing spans), we create dedicated training 
  examples for each span with surrounding context
- This preserves ALL spans and actually increases training data
- Each span gets focused attention during training

Usage:
    processor = AssetDeficitDataProcessor()
    examples = processor.process_json_file('data.json')
    train_loader = processor.create_data_loader(examples)
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class SpanAnnotation:
    """Represents a span annotation with its position and label."""
    start_char: int
    end_char: int
    label: str
    text: str
    confidence: float = 1.0

@dataclass
class TrainingExample:
    """Represents a training example for BERT token classification."""
    text: str
    tokens: List[str]
    labels: List[str]
    token_positions: List[Tuple[int, int]]
    metadata: Dict[str, Any]

class AssetDeficitDataProcessor:
    """
    Enhanced data processor that handles long documents via span-focused approach.
    
    Key Features:
    - Solves 512 token limit by creating span-focused training examples
    - Preserves all span annotations (no data loss)
    - Increases training data by creating multiple examples per document
    - Handles BIO tagging scheme for token classification
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 context_window: int = 200):
        """
        Initialize the data processor.
        
        Args:
            model_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length for BERT
            context_window: Characters to include around each span for context
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.context_window = context_window
        
        # Label mappings for BIO tagging
        self.label_to_id = {
            'O': 0,
            'B-ASSET': 1,
            'I-ASSET': 2,
            'B-DEFICIT': 3,
            'I-DEFICIT': 4
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        logger.info(f"Initialized processor with tokenizer: {model_name}")
        logger.info(f"Max length: {max_length}, Context window: {context_window}")
    
    def process_json_file(self, json_file: str) -> List[TrainingExample]:
        """
        Process a JSON file containing asset/deficit annotations.
        
        Args:
            json_file: Path to JSON file with document annotations
            
        Returns:
            List of training examples ready for BERT fine-tuning
        """
        logger.info(f"Processing JSON file: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_examples = []
        documents = data.get('data', [])
        documents_processed = 0
        documents_discarded = 0
        documents_no_spans = 0
        
        logger.info(f"Found {len(documents)} documents")
        
        for doc_idx, doc_data in enumerate(documents):
            doc_text = doc_data.get('extracted_content', '')
            if not doc_text:
                logger.warning(f"Document {doc_idx} has no extracted_content")
                documents_discarded += 1
                continue
                
            # Extract span data from linguistic_classification.analysis
            linguistic_data = doc_data.get('linguistic_classification', {})
            span_data = linguistic_data.get('analysis', [])
            
            # Create training examples directly from spans (no need to parse spans first)
            doc_examples = self._parse_spans(doc_text, span_data, doc_idx)
            
            if doc_examples:
                all_examples.extend(doc_examples)
                documents_processed += 1
            else:
                documents_no_spans += 1
        
        logger.info(f"Processing Summary:")
        logger.info(f"  - Documents with valid spans: {documents_processed}")
        logger.info(f"  - Documents discarded (no content): {documents_discarded}")
        logger.info(f"  - Documents discarded (no valid spans): {documents_no_spans}")
        logger.info(f"  - Total documents processed: {len(documents)}")
        logger.info(f"Created {len(all_examples)} training examples from {documents_processed} valid documents")
        self._log_dataset_statistics(all_examples)
        
        # Automatically save enhanced processing statistics
        try:
            stats_path = self.save_processing_stats()
            logger.info(f"Enhanced processing statistics saved to: {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save processing statistics: {e}")
        
        return all_examples
    
    def _parse_spans(self, text: str, span_data: List[Dict], doc_idx: int) -> List[TrainingExample]:
        """
        Parse span annotations and create training examples directly from text_spans.
        Since text_spans are already manageable, we create one training example per span.
        """
        examples = []
        
        for span_idx, span_info in enumerate(span_data):
            # Handle inconsistent data format: skip if span_info is not a dictionary
            if not isinstance(span_info, dict):
                logger.warning(f"Document {doc_idx}, span {span_idx}: Expected dict but got {type(span_info).__name__}: {str(span_info)[:100]}... Skipping.")
                continue
                
            text_span = span_info.get('text_span', '')
            span_type = span_info.get('type', '').upper()
            
            # Skip if no text span is provided
            if not text_span:
                continue
            
            # Get semantic phrases for this specific text_span
            semantic_assets = span_info.get('semantic_asset_phrases', [])
            semantic_deficits = span_info.get('semantic_deficit_phrases', [])
            
            # Create training example from this text_span
            example = self._create_text_span_example(
                text_span, span_type, semantic_assets, semantic_deficits, 
                doc_idx, span_idx
            )
            
            if example:
                examples.append(example)
                
        return examples
    
    def _create_text_span_example(self, text_span: str, span_type: str, 
                                 semantic_assets: List[str], semantic_deficits: List[str],
                                 doc_idx: int, span_idx: int) -> Optional[TrainingExample]:
        """
        Create a training example from a single text_span with its semantic phrases.
        """
        try:
            # Find positions of semantic phrases within the text_span
            span_annotations = []
            
            # Add semantic asset phrases
            for phrase in semantic_assets:
                if phrase and phrase.strip():
                    match_result = self._find_span_position(text_span, phrase.strip())
                    if match_result is not None:
                        start_pos, end_pos, match_type = match_result
                        span_annotations.append(SpanAnnotation(
                            start_char=start_pos,
                            end_char=end_pos,
                            label='ASSET',
                            text=text_span[start_pos:end_pos]
                        ))
                        if match_type != 'exact':
                            logger.debug(f"Found asset phrase using {match_type}: '{phrase}' in text_span")
                    else:
                        logger.debug(f"Could not find asset phrase '{phrase}' in text_span")
            
            # Add semantic deficit phrases  
            for phrase in semantic_deficits:
                if phrase and phrase.strip():
                    match_result = self._find_span_position(text_span, phrase.strip())
                    if match_result is not None:
                        start_pos, end_pos, match_type = match_result
                        span_annotations.append(SpanAnnotation(
                            start_char=start_pos,
                            end_char=end_pos,
                            label='DEFICIT',
                            text=text_span[start_pos:end_pos]
                        ))
                        if match_type != 'exact':
                            logger.debug(f"Found deficit phrase using {match_type}: '{phrase}' in text_span")
                    else:
                        logger.debug(f"Could not find deficit phrase '{phrase}' in text_span")
            
            # Remove overlapping spans
            span_annotations = self._remove_overlapping_spans(span_annotations)
            
            # Tokenize the text_span
            encoding = self.tokenizer(
                text_span,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            offset_mapping = encoding['offset_mapping']
            
            # Create BIO labels
            labels = self._create_bio_labels(span_annotations, offset_mapping, text_span)
            
            return TrainingExample(
                text=text_span,
                tokens=tokens,
                labels=labels,
                token_positions=offset_mapping,
                metadata={
                    'doc_id': doc_idx,
                    'span_id': span_idx,
                    'method': 'text_span_direct',
                    'span_type': span_type,
                    'num_asset_phrases': len(semantic_assets),
                    'num_deficit_phrases': len(semantic_deficits),
                    'num_found_spans': len(span_annotations),
                    'text_span_length': len(text_span),
                    'token_count': len(tokens),
                    'was_truncated': len(tokens) >= self.max_length
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating text span example for doc {doc_idx}, span {span_idx}: {e}")
            return None
    
    def _remove_overlapping_spans(self, spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        """Remove overlapping spans, keeping the longer ones or the first one if same length."""
        if not spans:
            return spans
        
        # Sort spans by start position
        spans.sort(key=lambda x: x.start_char)
        
        non_overlapping = []
        for current_span in spans:
            # Check if current span overlaps with any in non_overlapping list
            overlaps = False
            for existing_span in non_overlapping:
                # Check for overlap
                if (current_span.start_char < existing_span.end_char and 
                    current_span.end_char > existing_span.start_char):
                    overlaps = True
                    # Keep the longer span, or if same length, keep the first one
                    current_len = current_span.end_char - current_span.start_char
                    existing_len = existing_span.end_char - existing_span.start_char
                    if current_len > existing_len:
                        # Replace existing with current
                        non_overlapping.remove(existing_span)
                        non_overlapping.append(current_span)
                    # If current is not longer, we keep the existing one (do nothing)
                    break
            
            if not overlaps:
                non_overlapping.append(current_span)
        
        return non_overlapping

    def _find_span_position(self, text: str, span_text: str) -> Optional[tuple]:
        """
        Find span position using multiple matching strategies.
        Returns (start_pos, end_pos, match_type) or None if not found.
        """
        # Strategy 1: Exact match
        start_pos = text.find(span_text)
        if start_pos != -1:
            return (start_pos, start_pos + len(span_text), 'exact')
        
        # Strategy 2: Case-insensitive match
        text_lower = text.lower()
        span_lower = span_text.lower()
        start_pos = text_lower.find(span_lower)
        if start_pos != -1:
            return (start_pos, start_pos + len(span_text), 'case_insensitive')
        
        # Strategy 3: Fuzzy match with sliding window
        return self._fuzzy_find_span(text, span_text)
    
    def _fuzzy_find_span(self, text: str, span_text: str, threshold: float = 0.80) -> Optional[tuple]:
        """
        Find span using fuzzy matching with sliding window.
        Returns (start_pos, end_pos, match_type) or None if not found.
        """
        from difflib import SequenceMatcher
        
        span_len = len(span_text)
        best_score = 0
        best_pos = None
        
        # Handle very short spans differently
        if span_len < 10:
            threshold = 0.90  # Higher threshold for short spans
        
        # Sliding window approach
        step_size = max(1, span_len // 10)  # Optimize for performance
        for i in range(0, len(text) - span_len + 1, step_size):
            window = text[i:i + span_len]
            score = SequenceMatcher(None, span_text.lower(), window.lower()).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_pos = (i, i + span_len)
        
        # If no good match found with exact length, try variable length matching
        if not best_pos and span_len > 20:
            # Try matching with some length tolerance for longer spans
            for length_var in [-5, -10, 5, 10]:
                adjusted_len = span_len + length_var
                if adjusted_len <= 0 or adjusted_len > len(text):
                    continue
                    
                for i in range(0, len(text) - adjusted_len + 1, step_size):
                    window = text[i:i + adjusted_len]
                    score = SequenceMatcher(None, span_text.lower(), window.lower()).ratio()
                    
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_pos = (i, i + adjusted_len)
        
        if best_pos:
            return (best_pos[0], best_pos[1], f'fuzzy_{best_score:.2f}')
        
        return None
    
    def _create_bio_labels(self, spans: List[SpanAnnotation], offset_mapping: List[Tuple[int, int]], text: str) -> List[str]:
        """Create BIO (Begin-Inside-Outside) labels for tokens."""
        labels = ['O'] * len(offset_mapping)
        
        for span in spans:
            span_start, span_end = span.start_char, span.end_char
            first_token = True
            
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens (CLS, SEP, PAD)
                if token_start == token_end == 0:
                    continue
                
                # Check if token overlaps with span
                if token_start < span_end and token_end > span_start:
                    if first_token:
                        labels[token_idx] = f"B-{span.label}"
                        first_token = False
                    else:
                        labels[token_idx] = f"I-{span.label}"
        
        return labels
    
    def _log_dataset_statistics(self, examples: List[TrainingExample]):
        """Log useful statistics about the processed dataset."""
        if not examples:
            return
        
        # Count examples by method
        method_counts = {}
        label_counts = {}
        total_tokens = 0
        text_span_lengths = []
        token_counts = []
        truncation_counts = {}
        
        for example in examples:
            method = example.metadata.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            # Track text_span lengths
            text_span_length = example.metadata.get('text_span_length', len(example.text))
            text_span_lengths.append(text_span_length)
            
            # Track token counts
            num_tokens = len(example.tokens)
            token_counts.append(num_tokens)
            
            for label in example.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                total_tokens += 1
        
        # Calculate text_span size statistics
        if text_span_lengths:
            text_span_stats = {
                'min': min(text_span_lengths),
                'max': max(text_span_lengths),
                'mean': np.mean(text_span_lengths),
                'median': np.median(text_span_lengths),
                'std': np.std(text_span_lengths),
                'percentile_25': np.percentile(text_span_lengths, 25),
                'percentile_75': np.percentile(text_span_lengths, 75),
                'percentile_90': np.percentile(text_span_lengths, 90),
                'percentile_95': np.percentile(text_span_lengths, 95),
                'percentile_99': np.percentile(text_span_lengths, 99)
            }
        
        # Calculate token count statistics
        if token_counts:
            token_stats = {
                'min': min(token_counts),
                'max': max(token_counts),
                'mean': np.mean(token_counts),
                'median': np.median(token_counts),
                'std': np.std(token_counts),
                'percentile_95': np.percentile(token_counts, 95),
                'percentile_99': np.percentile(token_counts, 99)
            }
        
        # Calculate truncation risk at different max_length values
        max_length_options = [128, 256, 512, 1024]
        truncation_analysis = {}
        for max_len in max_length_options:
            truncated_count = sum(1 for count in token_counts if count > max_len)
            truncation_analysis[max_len] = {
                'would_truncate': truncated_count,
                'percentage': (truncated_count / len(token_counts)) * 100 if token_counts else 0
            }
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Processing methods used:")
        for method, count in method_counts.items():
            logger.info(f"    {method}: {count} examples")
        
        logger.info(f"  Label distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_tokens) * 100
            logger.info(f"    {label}: {count} ({percentage:.1f}%)")
        
        avg_tokens = total_tokens / len(examples)
        logger.info(f"  Average tokens per example: {avg_tokens:.1f}")
        
        # Log text_span size analysis
        if text_span_lengths:
            logger.info(f"  Text Span Length Analysis (characters):")
            logger.info(f"    Min: {text_span_stats['min']}")
            logger.info(f"    Max: {text_span_stats['max']}")
            logger.info(f"    Mean: {text_span_stats['mean']:.1f}")
            logger.info(f"    Median: {text_span_stats['median']:.1f}")
            logger.info(f"    95th percentile: {text_span_stats['percentile_95']:.1f}")
            logger.info(f"    99th percentile: {text_span_stats['percentile_99']:.1f}")
        
        # Log token count analysis  
        if token_counts:
            logger.info(f"  Token Count Analysis:")
            logger.info(f"    Min: {token_stats['min']}")
            logger.info(f"    Max: {token_stats['max']}")
            logger.info(f"    Mean: {token_stats['mean']:.1f}")
            logger.info(f"    95th percentile: {token_stats['percentile_95']:.1f}")
            logger.info(f"    99th percentile: {token_stats['percentile_99']:.1f}")
        
        # Log truncation risk analysis
        logger.info(f"  Truncation Risk Analysis:")
        for max_len, analysis in truncation_analysis.items():
            logger.info(f"    max_length={max_len}: {analysis['would_truncate']} examples ({analysis['percentage']:.1f}%) would be truncated")
        
        # Store stats for saving to JSON
        self._current_stats = {
            'total_examples': len(examples),
            'total_tokens': total_tokens,
            'label_distribution': label_counts,
            'processing_config': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'context_window': self.context_window
            },
            'example_types': method_counts,
            'text_span_analysis': text_span_stats if text_span_lengths else {},
            'token_analysis': token_stats if token_counts else {},
            'truncation_risk': truncation_analysis
        }
    
    def save_processing_stats(self, output_path: str = None) -> str:
        """
        Save detailed processing statistics to a JSON file.
        
        Args:
            output_path: Path to save the stats JSON file. If None, uses default location.
            
        Returns:
            Path to the saved statistics file
            
        Raises:
            RuntimeError: If no statistics are available (call process_json_file first)
        """
        if not hasattr(self, '_current_stats'):
            raise RuntimeError("No statistics available. Process data first using process_json_file().")
        
        if output_path is None:
            # Default to bert_finetuning/data/processed/processing_stats.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Go up from config to bert_finetuning
            output_path = os.path.join(project_root, 'data', 'processed', 'processing_stats.json')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        stats_for_json = self._convert_numpy_types(self._current_stats)
        
        # Add metadata about when the stats were generated
        from datetime import datetime
        stats_for_json['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'data_processor_version': '2.0_enhanced',
            'stats_format_version': '2.0'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing statistics saved to: {output_path}")
        return output_path
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    def create_data_loader(self, examples: List[TrainingExample], 
                          batch_size: int = 16, shuffle: bool = True, 
                          max_length: Optional[int] = None) -> DataLoader:
        """
        Create a PyTorch DataLoader from training examples.
        
        Args:
            examples: List of training examples
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            max_length: Maximum sequence length for padding. If None, uses the processor's max_length.
                       Can be different from processing max_length for optimization.
        """
        # Use provided max_length or fall back to processor's max_length
        dataloader_max_length = max_length if max_length is not None else self.max_length
        
        # Validation: training max_length shouldn't exceed processing max_length
        if dataloader_max_length > self.max_length:
            logger.warning(f"DataLoader max_length ({dataloader_max_length}) > processing max_length ({self.max_length}). "
                          f"This may cause truncation issues. Consider reprocessing data with max_length >= {dataloader_max_length}")
        
        # Log optimization opportunity
        if dataloader_max_length < self.max_length:
            logger.info(f"Using optimized DataLoader max_length: {dataloader_max_length} (processing was {self.max_length})")
        
        dataset = AssetDeficitDataset(examples, self.tokenizer, self.label_to_id, dataloader_max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch):
        """Custom collate function for batching examples."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def split_data(self, examples: List[TrainingExample], val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Split examples into train and validation sets using random shuffling.
        
        Args:
            examples: List of training examples
            val_ratio: Ratio of examples to use for validation (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        import random
        random.seed(random_seed)  # For reproducibility
        
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - val_ratio))
        train_examples = shuffled[:split_idx]
        val_examples = shuffled[split_idx:]
        
        logger.info(f"Data split (random with seed {random_seed}): {len(train_examples)} train, {len(val_examples)} validation")
        logger.info(f"Validation ratio: {val_ratio:.2f} ({len(val_examples)}/{len(examples)} examples)")
        
        return train_examples, val_examples
    
    def load_processed_examples(self, examples_data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """Load training examples from processed data."""
        examples = []
        for data in examples_data:
            examples.append(TrainingExample(
                text=data['text'],
                tokens=data['tokens'],
                labels=data['labels'],
                token_positions=data.get('token_positions', []),
                metadata=data.get('metadata', {})
            ))
        return examples
    
    def get_label_statistics(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Get statistics about label distribution."""
        label_counts = {}
        for example in examples:
            for label in example.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

class AssetDeficitDataset(Dataset):
    """PyTorch Dataset for asset/deficit classification."""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, label_to_id: Dict[str, int], max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Re-tokenize with proper padding and attention mask
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Create padded labels array
        labels = [-100] * self.max_length  # -100 is ignored by PyTorch loss
        
        # Map example labels to token positions
        for i, label in enumerate(example.labels):
            if i < self.max_length:
                labels[i] = self.label_to_id.get(label, 0)  # Default to 'O' if unknown
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        """Create a training example focused on a specific span."""
        # Extract context around the span
        start_context = max(0, span.start_char - self.context_window)
        end_context = min(len(content), span.end_char + self.context_window)
        
        context_text = content[start_context:end_context]
        
        # Adjust span positions relative to context
        adjusted_start = span.start_char - start_context
        adjusted_end = span.end_char - start_context
        
        # Tokenize
        tokenized = self.tokenizer(
            context_text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        tokens = tokenized['input_ids']
        offsets = tokenized['offset_mapping']
        
        # Create BIO labels
        labels = ['O'] * len(tokens)
        
        # Find tokens that overlap with the span and assign BIO tags correctly
        first_token_found = False
        for i, (start_offset, end_offset) in enumerate(offsets):
            if start_offset is None or end_offset is None:
                continue
                
            # Check if token overlaps with span
            if (start_offset < adjusted_end and end_offset > adjusted_start):
                if not first_token_found:
                    # First token of the span gets B- tag
                    labels[i] = f'B-{span.label}'
                    first_token_found = True
                else:
                    # Subsequent tokens get I- tag
                    labels[i] = f'I-{span.label}'
        
        # Convert to readable tokens
        readable_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        
        return TrainingExample(
            text=context_text,
            tokens=readable_tokens,
            labels=labels,
            token_positions=offsets,
            metadata={
                'document_id': doc.get('Sr', 'unknown'),
                'source': doc.get('source', ''),
                'span_text': span.text,
                'span_label': span.label,
                'example_type': 'span_focused'
            }
        )
    
    def _create_full_document_example(self, content: str, spans: List[SpanAnnotation], doc: Dict[str, Any]) -> Optional[TrainingExample]:
        """Create a training example for the full document."""
        # Tokenize full content
        tokenized = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        tokens = tokenized['input_ids']
        offsets = tokenized['offset_mapping']
        
        # Create BIO labels
        labels = ['O'] * len(tokens)
        
        # Sort spans by start position to handle overlaps properly
        sorted_spans = sorted(spans, key=lambda x: x.start_char)
        
        # Apply all spans with proper BIO tagging
        for span in sorted_spans:
            first_token_found = False
            for i, (start_offset, end_offset) in enumerate(offsets):
                if start_offset is None or end_offset is None:
                    continue
                    
                # Check if token overlaps with span
                if (start_offset < span.end_char and end_offset > span.start_char):
                    if not first_token_found:
                        # First token of this span gets B- tag
                        labels[i] = f'B-{span.label}'
                        first_token_found = True
                    else:
                        # Subsequent tokens get I- tag (only if not already labeled with different span)
                        if labels[i] == 'O' or labels[i].startswith('I-'):
                            labels[i] = f'I-{span.label}'
        
        # Convert to readable tokens
        readable_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        
        return TrainingExample(
            text=content[:self.max_length * 4],  # Rough truncation
            tokens=readable_tokens,
            labels=labels,
            token_positions=offsets,
            metadata={
                'document_id': doc.get('Sr', 'unknown'),
                'source': doc.get('source', ''),
                'span_count': len(spans),
                'example_type': 'full_document'
            }
        )
    
    def _create_no_span_example(self, content: str, doc: Dict[str, Any]) -> TrainingExample:
        """Create a training example with no spans (all O labels)."""
        # Tokenize content
        tokenized = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        tokens = tokenized['input_ids']
        offsets = tokenized['offset_mapping']
        
        # All O labels
        labels = ['O'] * len(tokens)
        
        # Convert to readable tokens
        readable_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        
        return TrainingExample(
            text=content[:self.max_length * 4],
            tokens=readable_tokens,
            labels=labels,
            token_positions=offsets,
            metadata={
                'document_id': doc.get('Sr', 'unknown'),
                'source': doc.get('source', ''),
                'example_type': 'no_spans'
            }
        )
    
    def create_data_loader(self, examples: List[TrainingExample], batch_size: int = 16, shuffle: bool = True, max_length: Optional[int] = None) -> DataLoader:
        """Create a PyTorch DataLoader from training examples.
        
        Args:
            examples: List of training examples
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            max_length: Optional max length for this specific DataLoader (defaults to processor's max_length)
        """
        effective_max_length = max_length if max_length is not None else self.max_length
        dataset = AssetDeficitDataset(examples, self.tokenizer, self.label_to_id, effective_max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['labels'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels)
        }
    
    def load_processed_examples(self, examples_data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """Load training examples from processed data."""
        examples = []
        for data in examples_data:
            examples.append(TrainingExample(
                text=data['text'],
                tokens=data['tokens'],
                labels=data['labels'],
                token_positions=data.get('token_positions', []),
                metadata=data.get('metadata', {})
            ))
        return examples
    
    def get_label_statistics(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Get statistics about label distribution."""
        label_counts = {}
        for example in examples:
            for label in example.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

class AssetDeficitDataset(Dataset):
    """PyTorch Dataset for asset/deficit classification."""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, label_to_id: Dict[str, int], max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize again to ensure consistency
        tokenized = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to ids, ensuring alignment
        label_ids = []
        tokens = self.tokenizer.tokenize(example.text)
        
        # Handle length differences due to special tokens
        effective_labels = example.labels[:self.max_length]
        
        for i in range(self.max_length):
            if i < len(effective_labels):
                label = effective_labels[i]
                label_ids.append(self.label_to_id.get(label, 0))
            else:
                label_ids.append(-100)  # Ignore in loss calculation
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }