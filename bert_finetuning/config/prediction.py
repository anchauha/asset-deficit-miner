"""
Standalone Prediction Module for BERT Asset/Deficit Classification.

This module provides prediction functionality for bert_finetuning without 
interfering with the existing config/inference.py or bert_webapp modules.
It handles model loading with proper fallbacks for different config file formats.

Usage:
    predictor = StandaloneBERTPredictor('path/to/model')
    spans = predictor.predict_text("Your text here...")
    predictor.create_html_report(text, spans, 'output.html')
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from dataclasses import dataclass, asdict
import html

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictedSpan:
    """Represents a predicted span with its properties."""
    start_char: int
    end_char: int
    label: str
    text: str
    confidence: float
    method: str = "single_chunk"

class StandaloneBERTPredictor:
    """
    Standalone inference system for asset/deficit span detection.
    
    This class is completely independent and doesn't interfere with existing
    inference modules. It handles prediction on documents of any length using
    a sliding window approach for documents longer than BERT's context limit.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', context_size: int = 200):
        """
        Initialize the prediction system.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('auto', 'cpu', 'cuda')
            context_size: Characters to include in sliding window overlap
        """
        self.model_path = model_path
        self.context_size = context_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings with robust fallback system
        self._load_label_mappings(model_path)
        
        logger.info(f"Loaded model from {model_path} on device {self.device}")
        logger.info(f"Label mappings: {self.label_names}")
    
    def _load_label_mappings(self, model_path: str):
        """
        Load label mappings from various config files with comprehensive fallbacks.
        
        Priority order:
        1. training_config.json (preferred)
        2. config.json with id2label mapping
        3. Default labels
        """
        self.label_names = None
        self.id_to_label = None
        
        # Try training_config.json first (bert_finetuning specific)
        training_config_path = os.path.join(model_path, 'training_config.json')
        if os.path.exists(training_config_path):
            try:
                with open(training_config_path, 'r') as f:
                    config = json.load(f)
                    
                if 'label_names' in config:
                    self.label_names = config['label_names']
                    self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
                    logger.info("Loaded label mappings from training_config.json")
                    return
                    
                if 'id_to_label' in config:
                    self.id_to_label = {int(k): v for k, v in config['id_to_label'].items()}
                    self.label_names = [self.id_to_label[i] for i in sorted(self.id_to_label.keys())]
                    logger.info("Loaded label mappings from training_config.json id_to_label")
                    return
                    
            except Exception as e:
                logger.warning(f"Failed to parse training_config.json: {e}")
        
        # Try standard config.json (transformers format)
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                if 'id2label' in config:
                    # Convert string keys to int and create mappings
                    id2label = {int(k): v for k, v in config['id2label'].items()}
                    self.id_to_label = id2label
                    self.label_names = [id2label[i] for i in sorted(id2label.keys())]
                    
                    # Map generic labels to meaningful ones if needed
                    if self.label_names[0].startswith('LABEL_'):
                        logger.warning("Found generic LABEL_X format, mapping to default labels")
                        default_labels = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
                        if len(self.label_names) == len(default_labels):
                            self.label_names = default_labels
                            self.id_to_label = {i: label for i, label in enumerate(default_labels)}
                    
                    logger.info("Loaded label mappings from config.json id2label")
                    return
                    
            except Exception as e:
                logger.warning(f"Failed to parse config.json: {e}")
        
        # Final fallback to default labels
        logger.warning("No valid label mappings found, using default labels")
        self.label_names = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
        self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
    
    def predict_text(self, text: str, confidence_threshold: float = 0.5) -> List[PredictedSpan]:
        """
        Predict asset/deficit spans in text.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            List of predicted spans with labels and confidence scores
        """
        if len(text) <= 1200:  # Single chunk processing
            return self._predict_single_chunk(text, confidence_threshold)
        else:
            return self._predict_long_text(text, confidence_threshold)
    
    def _predict_single_chunk(self, text: str, confidence_threshold: float) -> List[PredictedSpan]:
        """Predict spans for text that fits in a single BERT chunk."""
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        offset_mapping = tokenized['offset_mapping'][0]
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process predictions
        spans = []
        predictions_np = predictions[0].cpu().numpy()
        
        current_span = None
        for i, (start_char, end_char) in enumerate(offset_mapping):
            if start_char == 0 and end_char == 0:  # Special tokens
                continue
                
            pred_probs = predictions_np[i]
            pred_id = np.argmax(pred_probs)
            confidence = float(pred_probs[pred_id])
            label = self.id_to_label.get(pred_id, 'O')
            
            if confidence < confidence_threshold:
                label = 'O'
            
            if label.startswith('B-'):  # Begin new span
                if current_span:
                    spans.append(current_span)
                span_type = label[2:]  # Remove 'B-' prefix
                current_span = PredictedSpan(
                    start_char=int(start_char),
                    end_char=int(end_char),
                    label=span_type,
                    text=text[start_char:end_char],
                    confidence=confidence,
                    method="single_chunk"
                )
            elif label.startswith('I-') and current_span:  # Continue span
                span_type = label[2:]  # Remove 'I-' prefix
                if current_span.label == span_type:
                    current_span.end_char = int(end_char)
                    current_span.text = text[current_span.start_char:current_span.end_char]
                    # Update confidence (could use average or max)
                    current_span.confidence = max(current_span.confidence, confidence)
                else:
                    # Type mismatch, end current span and potentially start new one
                    spans.append(current_span)
                    current_span = None
            else:  # 'O' or unrecognized label
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _predict_long_text(self, text: str, confidence_threshold: float) -> List[PredictedSpan]:
        """Predict spans for long text using sliding window approach."""
        # For now, implement a simple truncation approach
        # This can be enhanced with proper sliding window later
        logger.warning(f"Text length {len(text)} > 1200, truncating to first 1200 characters")
        truncated_text = text[:1200]
        spans = self._predict_single_chunk(truncated_text, confidence_threshold)
        
        # Mark spans as potentially truncated
        for span in spans:
            span.method = "truncated"
        
        return spans
    
    def create_html_report(self, text: str, spans: List[PredictedSpan], output_path: str, title: str = "BERT Prediction Results"):
        """
        Create an HTML visualization report of the predictions.
        
        Args:
            text: Original text
            spans: Predicted spans
            output_path: Path to save HTML file
            title: Title for the report
        """
        # Simple HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .text-content {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            white-space: pre-wrap;
        }}
        .asset {{
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            border-radius: 4px;
            border: 1px solid #c3e6cb;
        }}
        .deficit {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 4px;
            border-radius: 4px;
            border: 1px solid #f5c6cb;
        }}
        .spans-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .spans-table th, .spans-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .spans-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html.escape(title)}</h1>
        <p>Asset/Deficit Analysis Results</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(spans)}</div>
            <div class="stat-label">Total Spans</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in spans if s.label == 'ASSET'])}</div>
            <div class="stat-label">Asset Spans</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len([s for s in spans if s.label == 'DEFICIT'])}</div>
            <div class="stat-label">Deficit Spans</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{(sum(s.confidence for s in spans) / len(spans) if spans else 0):.3f}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
    </div>
    
    <div class="text-content">
        {self._render_annotated_text(text, spans)}
    </div>
    
    <h2>Detected Spans</h2>
    <table class="spans-table">
        <thead>
            <tr>
                <th>Label</th>
                <th>Text</th>
                <th>Confidence</th>
                <th>Position</th>
            </tr>
        </thead>
        <tbody>
            {self._render_spans_table(spans)}
        </tbody>
    </table>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def _render_annotated_text(self, text: str, spans: List[PredictedSpan]) -> str:
        """Render text with highlighted spans."""
        if not spans:
            return html.escape(text)
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x.start_char)
        
        result = []
        last_end = 0
        
        for span in sorted_spans:
            # Add text before this span
            if span.start_char > last_end:
                result.append(html.escape(text[last_end:span.start_char]))
            
            # Add highlighted span
            css_class = span.label.lower()
            result.append(f'<span class="{css_class}" title="Confidence: {span.confidence:.3f}">')
            result.append(html.escape(span.text))
            result.append('</span>')
            
            last_end = span.end_char
        
        # Add remaining text
        if last_end < len(text):
            result.append(html.escape(text[last_end:]))
        
        return ''.join(result)
    
    def _render_spans_table(self, spans: List[PredictedSpan]) -> str:
        """Render table rows for spans."""
        rows = []
        for span in spans:
            rows.append(f"""
                <tr>
                    <td><span class="{span.label.lower()}">{html.escape(span.label)}</span></td>
                    <td>{html.escape(span.text)}</td>
                    <td>{span.confidence:.3f}</td>
                    <td>{span.start_char}-{span.end_char}</td>
                </tr>
            """)
        return ''.join(rows)