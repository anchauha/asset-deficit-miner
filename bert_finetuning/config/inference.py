"""
Inference System for Asset/Deficit BERT Classification.

This module provides inference capabilities for a trained BERT model and
generates a modern, interactive HTML visualization of the results using
Tailwind CSS and JavaScript.

Key Features:
- Handles documents longer than 512 tokens using a sliding window approach.
- Generates a modern, interactive HTML report with filtering and dark mode.
- Serializes span data to JSON, cleanly separating data from presentation.
- Provides confidence scores for all predictions.

Usage:
    inference = AssetDeficitInference('path/to/model')
    spans = inference.predict_text("Your document text here...")
    
    visualizer = TextVisualizer()
    html_content = visualizer.create_html_visualization(text, spans)
    with open("analysis_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
from typing import List, Dict, Any
import logging
from dataclasses import dataclass, asdict
import html
import os

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictedSpan:
    """Represents a predicted span with confidence score."""
    start_char: int
    end_char: int
    label: str
    text: str
    confidence: float
    token_scores: List[float]

class AssetDeficitInference:
    """
    Inference system for trained BERT asset/deficit classification model.
    
    Handles documents of any length using a sliding window approach to
    overcome the 512 token limitation.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the inference system.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'label_names': ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT'],
                'max_length': 512
            }
        
        self.label_names = self.config['label_names']
        self.max_length = self.config.get('max_length', 512)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path} on device {self.device}")
        logger.info(f"Labels: {self.label_names}")
    
    def predict_text(self, text: str, confidence_threshold: float = 0.5) -> List[PredictedSpan]:
        """
        Predict asset/deficit spans in text.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence for span prediction
            
        Returns:
            List of predicted spans
        """
        if len(text.strip()) == 0:
            return []
        
        # Use sliding window for long texts
        if len(text) > self.max_length * 3:  # Rough character estimate
            return self._predict_sliding_window(text, confidence_threshold)
        else:
            return self._predict_single_window(text, confidence_threshold)
    
    def _predict_single_window(self, text: str, confidence_threshold: float) -> List[PredictedSpan]:
        """Predict spans for a single window of text."""
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        offset_mapping = tokenized['offset_mapping'][0]
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Convert predictions to spans
        predicted_labels = predicted_labels[0].cpu().numpy()
        confidences = predictions[0].cpu().numpy()
        
        spans = self._extract_spans_from_predictions(
            text, predicted_labels, confidences, offset_mapping, confidence_threshold
        )
        
        return spans
    
    def _predict_sliding_window(self, text: str, confidence_threshold: float, window_size: int = 400, overlap: int = 50) -> List[PredictedSpan]:
        """Predict spans using sliding window approach for long texts."""
        all_spans = []
        
        start = 0
        while start < len(text):
            end = min(start + window_size, len(text))
            window_text = text[start:end]
            
            # Predict for this window
            window_spans = self._predict_single_window(window_text, confidence_threshold)
            
            # Adjust span positions to global coordinates
            for span in window_spans:
                span.start_char += start
                span.end_char += start
                span.text = text[span.start_char:span.end_char]
            
            all_spans.extend(window_spans)
            
            # Move window
            if end >= len(text):
                break
            start = end - overlap
        
        # Merge overlapping spans
        merged_spans = self._merge_overlapping_spans(all_spans)
        
        return merged_spans
    
    def _extract_spans_from_predictions(self, text: str, predicted_labels: np.ndarray, 
                                      confidences: np.ndarray, offset_mapping: List, 
                                      confidence_threshold: float) -> List[PredictedSpan]:
        """Extract spans from token predictions."""
        spans = []
        current_span = None
        
        for i, (label_id, token_confidences) in enumerate(zip(predicted_labels, confidences)):
            if i >= len(offset_mapping) or offset_mapping[i][0] is None:
                continue
                
            label = self.label_names[label_id]
            confidence = float(np.max(token_confidences))
            
            start_offset, end_offset = offset_mapping[i]
            
            if label.startswith('B-'):  # Beginning of span
                # End previous span if exists
                if current_span:
                    spans.append(current_span)
                
                # Start new span
                span_type = label[2:]  # Remove 'B-'
                current_span = {
                    'start_char': int(start_offset),
                    'end_char': int(end_offset),
                    'label': span_type,
                    'confidences': [confidence],
                    'token_count': 1
                }
                
            elif label.startswith('I-') and current_span:  # Inside span
                span_type = label[2:]  # Remove 'I-'
                if current_span['label'] == span_type:
                    # Extend current span
                    current_span['end_char'] = int(end_offset)
                    current_span['confidences'].append(confidence)
                    current_span['token_count'] += 1
                else:
                    # Different span type, end current and start new
                    spans.append(self._finalize_span(current_span, text))
                    current_span = {
                        'start_char': int(start_offset),
                        'end_char': int(end_offset),
                        'label': span_type,
                        'confidences': [confidence],
                        'token_count': 1
                    }
            else:  # 'O' or mismatch
                # End current span
                if current_span:
                    spans.append(self._finalize_span(current_span, text))
                    current_span = None
        
        # End final span
        if current_span:
            spans.append(self._finalize_span(current_span, text))
        
        # Filter by confidence threshold
        filtered_spans = []
        for span in spans:
            if span.confidence >= confidence_threshold:
                filtered_spans.append(span)
        
        return filtered_spans
    
    def _finalize_span(self, span_dict: Dict, text: str) -> PredictedSpan:
        """Convert span dictionary to PredictedSpan object."""
        avg_confidence = float(np.mean(span_dict['confidences']))
        span_text = text[span_dict['start_char']:span_dict['end_char']]
        
        return PredictedSpan(
            start_char=span_dict['start_char'],
            end_char=span_dict['end_char'],
            label=span_dict['label'],
            text=span_text,
            confidence=avg_confidence,
            token_scores=span_dict['confidences']
        )
    
    def _merge_overlapping_spans(self, spans: List[PredictedSpan]) -> List[PredictedSpan]:
        """Merge overlapping spans from sliding window prediction."""
        if not spans:
            return []
        
        # Sort spans by start position
        spans.sort(key=lambda x: x.start_char)
        
        merged = []
        current = spans[0]
        
        for next_span in spans[1:]:
            # Check for overlap
            if (next_span.start_char <= current.end_char and 
                next_span.label == current.label):
                # Merge spans
                current.end_char = max(current.end_char, next_span.end_char)
                current.confidence = max(current.confidence, next_span.confidence)
                current.token_scores.extend(next_span.token_scores)
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_span
        
        merged.append(current)
        
        # Update text for merged spans
        for span in merged:
            span.text = span.text  # Text should be updated by caller
        
        return merged

class TextVisualizer:
    """
    Creates interactive HTML visualizations of asset/deficit analysis results.
    """
    
    def create_html_visualization(self, text: str, spans: List[PredictedSpan], 
                                title: str = "Asset/Deficit Analysis") -> str:
        """
        Create an interactive HTML visualization.
        
        Args:
            text: Original text
            spans: Predicted spans
            title: Page title
            
        Returns:
            HTML content as string
        """
        # Prepare data
        spans_data = [asdict(span) for span in spans]
        
        # Generate HTML
        html_content = HTML_TEMPLATE.format(
            title=html.escape(title),
            text_content=html.escape(text),
            spans_data=json.dumps(spans_data, indent=2),
            stats=self._generate_stats(spans)
        )
        
        return html_content
    
    def _generate_stats(self, spans: List[PredictedSpan]) -> str:
        """Generate statistics HTML."""
        if not spans:
            return "<p>No spans detected.</p>"
        
        # Count by label
        label_counts = {}
        total_confidence = 0
        
        for span in spans:
            label_counts[span.label] = label_counts.get(span.label, 0) + 1
            total_confidence += span.confidence
        
        avg_confidence = total_confidence / len(spans) if spans else 0
        
        stats_html = f"""
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
                <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">{len(spans)}</div>
                <div class="text-sm text-gray-600 dark:text-gray-400">Total Spans</div>
            </div>
            <div class="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
                <div class="text-2xl font-bold text-green-600 dark:text-green-400">{label_counts.get('ASSET', 0)}</div>
                <div class="text-sm text-gray-600 dark:text-gray-400">Asset Spans</div>
            </div>
            <div class="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
                <div class="text-2xl font-bold text-red-600 dark:text-red-400">{label_counts.get('DEFICIT', 0)}</div>
                <div class="text-sm text-gray-600 dark:text-gray-400">Deficit Spans</div>
            </div>
            <div class="bg-purple-50 dark:bg-purple-900 p-4 rounded-lg">
                <div class="text-2xl font-bold text-purple-600 dark:text-purple-400">{avg_confidence:.2f}</div>
                <div class="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</div>
            </div>
        </div>
        """
        
        return stats_html

# HTML Template for visualization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
        }}
        .span-asset {{
            background-color: #dcfce7;
            border: 1px solid #16a34a;
            border-radius: 4px;
            padding: 2px 4px;
            margin: 0 1px;
            cursor: pointer;
        }}
        .span-deficit {{
            background-color: #fef2f2;
            border: 1px solid #dc2626;
            border-radius: 4px;
            padding: 2px 4px;
            margin: 0 1px;
            cursor: pointer;
        }}
        .dark .span-asset {{
            background-color: #14532d;
            border-color: #22c55e;
        }}
        .dark .span-deficit {{
            background-color: #7f1d1d;
            border-color: #ef4444;
        }}
        .span-hidden {{
            background-color: transparent !important;
            border: none !important;
        }}
    </style>
</head>
<body class="bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200">
    <div class="container mx-auto p-6 max-w-6xl">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">{title}</h1>
            <button id="darkModeToggle" class="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path class="dark:hidden" d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
                    <path class="hidden dark:block" d="M10 2L9 9l7-7-7 7-7-7 7 7-7 7 7-7 7 7-7-7 7-7-7 7V2z"></path>
                </svg>
            </button>
        </div>

        <!-- Statistics -->
        {stats}

        <!-- Controls -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Controls</h2>
            <div class="flex flex-wrap gap-4">
                <label class="flex items-center">
                    <input type="checkbox" id="showAsset" checked class="mr-2">
                    <span class="text-green-600 dark:text-green-400">Asset Spans</span>
                </label>
                <label class="flex items-center">
                    <input type="checkbox" id="showDeficit" checked class="mr-2">
                    <span class="text-red-600 dark:text-red-400">Deficit Spans</span>
                </label>
                <div class="flex items-center">
                    <label class="mr-2">Confidence Threshold:</label>
                    <input type="range" id="confidenceThreshold" min="0" max="1" step="0.1" value="0.5" class="mr-2">
                    <span id="confidenceValue">0.5</span>
                </div>
            </div>
        </div>

        <!-- Text Analysis -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Text Analysis</h2>
            <div id="analyzedText" class="text-lg leading-relaxed border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-gray-50 dark:bg-gray-900">
                <!-- Text with highlighted spans will be inserted here -->
            </div>
        </div>

        <!-- Span Details -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold mb-4">Detected Spans</h2>
            <div id="spansList" class="space-y-3">
                <!-- Span details will be inserted here -->
            </div>
        </div>
    </div>

    <script>
        // Data
        const originalText = `{text_content}`;
        const spansData = {spans_data};

        // Dark mode toggle
        const darkModeToggle = document.getElementById('darkModeToggle');
        const html = document.documentElement;

        darkModeToggle.addEventListener('click', () => {{
            html.classList.toggle('dark');
        }});

        // Initialize visualization
        function initializeVisualization() {{
            updateTextVisualization();
            updateSpansList();
            setupEventListeners();
        }}

        function updateTextVisualization() {{
            const showAsset = document.getElementById('showAsset').checked;
            const showDeficit = document.getElementById('showDeficit').checked;
            const confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value);

            let html = originalText;
            const spans = spansData
                .filter(span => span.confidence >= confidenceThreshold)
                .filter(span => (span.label === 'ASSET' && showAsset) || (span.label === 'DEFICIT' && showDeficit))
                .sort((a, b) => b.start_char - a.start_char); // Reverse order for insertion

            spans.forEach((span, index) => {{
                const spanClass = span.label === 'ASSET' ? 'span-asset' : 'span-deficit';
                const beforeText = html.substring(0, span.start_char);
                const spanText = html.substring(span.start_char, span.end_char);
                const afterText = html.substring(span.end_char);
                
                const wrappedSpan = `<span class="${{spanClass}}" data-span-id="${{index}}" title="Confidence: ${{span.confidence.toFixed(3)}}">${{spanText}}</span>`;
                html = beforeText + wrappedSpan + afterText;
            }});

            document.getElementById('analyzedText').innerHTML = html;
        }}

        function updateSpansList() {{
            const showAsset = document.getElementById('showAsset').checked;
            const showDeficit = document.getElementById('showDeficit').checked;
            const confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value);

            const filteredSpans = spansData
                .filter(span => span.confidence >= confidenceThreshold)
                .filter(span => (span.label === 'ASSET' && showAsset) || (span.label === 'DEFICIT' && showDeficit));

            const spansList = document.getElementById('spansList');
            spansList.innerHTML = '';

            filteredSpans.forEach((span, index) => {{
                const spanElement = document.createElement('div');
                spanElement.className = 'border border-gray-200 dark:border-gray-700 rounded-lg p-4';
                
                const labelColor = span.label === 'ASSET' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
                
                spanElement.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-lg font-semibold ${{labelColor}}">${{span.label}}</span>
                        <span class="text-sm text-gray-500 dark:text-gray-400">Confidence: ${{span.confidence.toFixed(3)}}</span>
                    </div>
                    <div class="text-gray-700 dark:text-gray-300 mb-2">"${{span.text}}"</div>
                    <div class="text-sm text-gray-500 dark:text-gray-400">
                        Position: ${{span.start_char}}-${{span.end_char}} | Length: ${{span.text.length}} chars
                    </div>
                `;
                
                spansList.appendChild(spanElement);
            }});

            if (filteredSpans.length === 0) {{
                spansList.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center">No spans match the current filters.</p>';
            }}
        }}

        function setupEventListeners() {{
            document.getElementById('showAsset').addEventListener('change', () => {{
                updateTextVisualization();
                updateSpansList();
            }});

            document.getElementById('showDeficit').addEventListener('change', () => {{
                updateTextVisualization();
                updateSpansList();
            }});

            const confidenceSlider = document.getElementById('confidenceThreshold');
            const confidenceValue = document.getElementById('confidenceValue');

            confidenceSlider.addEventListener('input', (e) => {{
                confidenceValue.textContent = e.target.value;
                updateTextVisualization();
                updateSpansList();
            }});
        }}

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeVisualization);
    </script>
</body>
</html>
"""