
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

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- HTML Template for the visualization report ---
# Data placeholders like %%TITLE%% will be replaced by the TextVisualizer class.
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%%TITLE%%</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                    },
                    colors: {
                        asset: {
                            light: '#dcfce7', // green-100
                            dark: '#166534',  // green-800
                            text: '#16a34a', // green-600
                        },
                        deficit: {
                            light: '#fee2e2', // red-100
                            dark: '#991b1b',  // red-800
                            text: '#ef4444', // red-500
                        },
                    },
                }
            },
            darkMode: 'class'
        }
    </script>
    <style>
        .highlight-span { transition: background-color 0.3s ease; }
    </style>
</head>
<body class="bg-slate-50 dark:bg-gray-900 text-slate-800 dark:text-slate-200">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8">

        <header class="text-center mb-8">
            <h1 class="text-3xl sm:text-4xl font-bold text-slate-900 dark:text-white">%%TITLE%%</h1>
            <p class="text-md text-slate-600 dark:text-slate-400 mt-2">BERT-based span detection.</p>
        </header>

        <main class="space-y-8">
            <div class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md border border-slate-200 dark:border-gray-700">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                    <div class="col-span-1">
                        <label for="filter" class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Filter by Label</label>
                        <div id="filter-buttons" class="flex space-x-2">
                            <button data-filter="ALL" class="filter-btn active flex-1 bg-blue-500 text-white px-4 py-2 rounded-md text-sm font-semibold">All</button>
                            <button data-filter="ASSET" class="filter-btn flex-1 bg-slate-200 dark:bg-gray-700 px-4 py-2 rounded-md text-sm font-semibold">Assets</button>
                            <button data-filter="DEFICIT" class="filter-btn flex-1 bg-slate-200 dark:bg-gray-700 px-4 py-2 rounded-md text-sm font-semibold">Deficits</button>
                        </div>
                    </div>
                    <div class="col-span-1">
                        <label for="confidence-slider" class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                            Confidence Threshold: <span id="confidence-value" class="font-bold">0.85</span>
                        </label>
                        <input id="confidence-slider" type="range" min="0.5" max="1.0" value="0.85" step="0.01" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700">
                    </div>
                    <div class="col-span-1 flex justify-center md:justify-end items-center">
                         <label for="darkModeToggle" class="mr-3 text-sm font-medium">Dark Mode</label>
                         <button id="darkModeToggle" class="relative inline-flex items-center h-6 rounded-full w-11 transition-colors bg-slate-200 dark:bg-gray-600 focus:outline-none">
                            <span class="inline-block w-4 h-4 transform bg-white rounded-full transition-transform translate-x-1 dark:translate-x-6"></span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md border border-slate-200 dark:border-gray-700">
                <h2 class="text-2xl font-bold mb-4 text-slate-900 dark:text-white">Annotated Document</h2>
                <p id="annotated-text" class="text-lg leading-relaxed whitespace-pre-wrap"></p>
            </div>
            
            <div id="stats-container"></div>

            <div class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md border border-slate-200 dark:border-gray-700">
                 <h2 class="text-2xl font-bold mb-4 text-slate-900 dark:text-white">Detected Spans</h2>
                 <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-slate-200 dark:divide-gray-700">
                        <thead class="bg-slate-50 dark:bg-gray-700">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-300 uppercase tracking-wider">Label</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-300 uppercase tracking-wider">Text</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-300 uppercase tracking-wider">Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="spans-table-body" class="bg-white dark:bg-gray-800 divide-y divide-slate-200 dark:divide-gray-700"></tbody>
                    </table>
                 </div>
            </div>
        </main>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const originalText = %%TEXT_JSON%%;
            const allSpans = %%SPANS_JSON%%;
            
            let currentFilter = 'ALL';
            let confidenceThreshold = 0.85;

            const annotatedTextEl = document.getElementById('annotated-text');
            const spansTableBodyEl = document.getElementById('spans-table-body');
            const filterButtonsEl = document.getElementById('filter-buttons');
            const confidenceSliderEl = document.getElementById('confidence-slider');
            const confidenceValueEl = document.getElementById('confidence-value');
            const darkModeToggle = document.getElementById('darkModeToggle');

            const colors = {
                ASSET: { light: 'bg-asset-light text-asset-text dark:bg-asset-dark dark:text-asset-light' },
                DEFICIT: { light: 'bg-deficit-light text-deficit-text dark:bg-deficit-dark dark:text-deficit-light' }
            };
            
            function escapeHTML(str) {
                const p = document.createElement('p');
                p.textContent = str;
                return p.innerHTML;
            }

            function render() {
                const filteredSpans = allSpans
                    .filter(span => currentFilter === 'ALL' || span.label === currentFilter)
                    .filter(span => span.confidence >= confidenceThreshold);
                
                renderAnnotatedText(filteredSpans);
                renderSpansTable(filteredSpans);
                updateStats(filteredSpans);
            }

            function renderAnnotatedText(spans) {
                let lastIndex = 0;
                const fragments = [];
                spans.forEach(span => {
                    if (span.start_char > lastIndex) {
                        fragments.push(document.createTextNode(originalText.slice(lastIndex, span.start_char)));
                    }
                    const originalIndex = allSpans.indexOf(span);
                    const spanEl = document.createElement('mark');
                    spanEl.id = `span-mark-${originalIndex}`;
                    spanEl.textContent = span.text;
                    spanEl.className = `highlight-span p-1 rounded-md cursor-pointer ${colors[span.label]?.light || ''}`;
                    spanEl.title = `Label: ${span.label} | Confidence: ${span.confidence.toFixed(3)}`;
                    spanEl.dataset.spanIndex = originalIndex;
                    fragments.push(spanEl);
                    lastIndex = span.end_char;
                });
                if (lastIndex < originalText.length) {
                    fragments.push(document.createTextNode(originalText.slice(lastIndex)));
                }
                annotatedTextEl.innerHTML = '';
                fragments.forEach(frag => annotatedTextEl.appendChild(frag));
            }

            function renderSpansTable(spans) {
                spansTableBodyEl.innerHTML = spans.map(span => {
                    const originalIndex = allSpans.indexOf(span);
                    return `
                    <tr id="span-row-${originalIndex}" data-span-index="${originalIndex}" class="hover:bg-slate-100 dark:hover:bg-gray-700 cursor-pointer">
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${colors[span.label]?.light || ''}">${span.label}</span>
                        </td>
                        <td class="px-6 py-4 whitespace-normal">${escapeHTML(span.text)}</td>
                        <td class="px-6 py-4 whitespace-nowrap font-mono text-sm">${span.confidence.toFixed(3)}</td>
                    </tr>`;
                }).join('');
            }

            function updateStats(spans) {
                 const stats = calculateStatistics(spans, originalText);
                 const statsEl = document.getElementById('stats-container');
                 const statCards = [
                    { label: 'Total Spans', value: stats.total_spans },
                    { label: 'Asset Spans', value: stats.asset_count },
                    { label: 'Deficit Spans', value: stats.deficit_count },
                    { label: 'Avg. Confidence', value: stats.avg_confidence.toFixed(3) },
                    { label: 'Text Coverage', value: `${stats.text_coverage.toFixed(1)}%` },
                    { label: 'Avg. Span Length', value: `${stats.avg_span_length.toFixed(0)} chars` }
                 ];
                 statsEl.innerHTML = `
                    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                        ${statCards.map(card => `
                            <div class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md text-center border border-slate-200 dark:border-gray-700">
                                <p class="text-2xl font-bold text-blue-600 dark:text-blue-400">${card.value}</p>
                                <p class="text-xs text-slate-500 dark:text-slate-400 uppercase font-semibold tracking-wider">${card.label}</p>
                            </div>`).join('')}
                    </div>`;
            }
            
            function calculateStatistics(spans, text) {
                if (!spans || spans.length === 0) {
                    return { total_spans: 0, asset_count: 0, deficit_count: 0, avg_confidence: 0, text_coverage: 0, avg_span_length: 0 };
                }
                const asset_count = spans.filter(s => s.label === 'ASSET').length;
                const deficit_count = spans.filter(s => s.label === 'DEFICIT').length;
                const avg_confidence = spans.reduce((sum, s) => sum + s.confidence, 0) / spans.length;
                const total_span_chars = spans.reduce((sum, s) => sum + (s.end_char - s.start_char), 0);
                const text_coverage = text.length > 0 ? (total_span_chars / text.length) * 100 : 0;
                const avg_span_length = total_span_chars / spans.length;
                return { total_spans: spans.length, asset_count, deficit_count, avg_confidence, text_coverage, avg_span_length };
            }

            filterButtonsEl.addEventListener('click', e => {
                if (e.target.tagName === 'BUTTON') {
                    currentFilter = e.target.dataset.filter;
                    document.querySelectorAll('.filter-btn').forEach(btn => {
                        btn.classList.remove('active', 'bg-blue-500', 'text-white');
                        btn.classList.add('bg-slate-200', 'dark:bg-gray-700');
                    });
                    e.target.classList.add('active', 'bg-blue-500', 'text-white');
                    e.target.classList.remove('bg-slate-200', 'dark:bg-gray-700');
                    render();
                }
            });

            confidenceSliderEl.addEventListener('input', e => {
                confidenceThreshold = parseFloat(e.target.value);
                confidenceValueEl.textContent = confidenceThreshold.toFixed(2);
                render();
            });
            
            document.body.addEventListener('click', e => {
                const target = e.target;
                const spanIndex = target.closest('[data-span-index]')?.dataset.spanIndex;
                if (spanIndex) {
                    document.querySelectorAll('.highlight-span, [id^="span-row-"]').forEach(el => el.classList.remove('ring-2', 'ring-blue-500'));
                    const markEl = document.getElementById(`span-mark-${spanIndex}`);
                    const rowEl = document.getElementById(`span-row-${spanIndex}`);
                    if(markEl) {
                        markEl.classList.add('ring-2', 'ring-blue-500');
                        markEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    if(rowEl) {
                        rowEl.classList.add('ring-2', 'ring-blue-500');
                        rowEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }
            });
            
            const ls = localStorage.getItem('darkMode');
            if (ls === 'true' || (!ls && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.classList.add('dark');
            }
            darkModeToggle.addEventListener('click', () => {
                const isDark = document.documentElement.classList.toggle('dark');
                localStorage.setItem('darkMode', isDark);
            });

            render();
        });
    </script>
</body>
</html>
"""

@dataclass
class PredictedSpan:
    """Represents a predicted span with its properties."""
    start_char: int
    end_char: int
    label: str
    text: str
    confidence: float
    method: str = "single_chunk"

class AssetDeficitInference:
    """
    Inference system for asset/deficit span detection.
    
    Handles prediction on documents of any length using a sliding window
    approach for documents longer than BERT's context limit.
    """
    
    def __init__(self, model_path: str, context_size: int = 200):
        """
        Initialize the inference system.
        
        Args:
            model_path: Path to the trained model directory
            context_size: Characters to include in sliding window overlap
        """
        self.model_path = model_path
        self.context_size = context_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except (OSError, FileNotFoundError, TypeError):
            logger.warning(f"Tokenizer not found in {model_path}, using bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self._load_label_mappings(model_path)
        logger.info(f"Loaded model from {model_path} on device {self.device}")
    
    def _load_label_mappings(self, model_path: str):
        """Load label mappings from training config."""
        try:
            with open(f"{model_path}/training_config.json", 'r') as f:
                config = json.load(f)
                
                # Try to get id_to_label from config, or create it from label_names
                if 'id_to_label' in config:
                    self.id_to_label = {int(k): v for k, v in config['id_to_label'].items()}
                elif 'label_names' in config:
                    self.label_names = config['label_names']
                    self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
                else:
                    # Fallback to default
                    self.label_names = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
                    self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
                
                # Ensure label_names is set
                if 'label_names' in config:
                    self.label_names = config['label_names']
                else:
                    # Create label_names from id_to_label if not already set
                    if not hasattr(self, 'label_names'):
                        self.label_names = [self.id_to_label[i] for i in sorted(self.id_to_label.keys())]
                        
        except FileNotFoundError:
            logger.warning("Training config not found, using default label mappings")
            self.label_names = ['O', 'B-ASSET', 'I-ASSET', 'B-DEFICIT', 'I-DEFICIT']
            self.id_to_label = {i: label for i, label in enumerate(self.label_names)}
    
    def predict_text(self, text: str) -> List[PredictedSpan]:
        """Predict asset/deficit spans in a text."""
        if len(text) <= 1200:  # Reduced from 400 to be safer with token limits
            return self._predict_single_chunk(text)
        return self._predict_sliding_window(text)
    
    def _predict_single_chunk(self, text: str) -> List[PredictedSpan]:
        """Predict spans in a single text chunk."""
        encoding = self.tokenizer(
            text, return_tensors='pt', return_offsets_mapping=True,
            truncation=True, max_length=512, padding=True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1)[0]
        
        return self._convert_predictions_to_spans(
            predictions.cpu().numpy(), confidences.cpu().numpy(),
            offset_mapping, text, method="single_chunk"
        )
    
    def _predict_sliding_window(self, text: str) -> List[PredictedSpan]:
        """Predict spans using sliding window for long texts."""
        window_size = 300  # Reduced window size for better token management
        overlap = 100      # Increased overlap to catch boundary spans
        all_spans = []
        seen_spans = set()
        
        for start in range(0, len(text), window_size - overlap):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]
            
            # Skip very small chunks at the end
            if len(chunk_text.strip()) < 20:
                continue
                
            chunk_spans = self._predict_single_chunk(chunk_text)
            
            for span in chunk_spans:
                global_start = span.start_char + start
                global_end = span.end_char + start
                
                # Ensure span is within text bounds
                if global_end > len(text):
                    global_end = len(text)
                    
                span_key = (global_start, global_end, span.label)
                
                # Check for overlapping spans and keep the one with higher confidence
                overlapping = False
                for existing_span in all_spans:
                    if (existing_span.start_char < global_end and 
                        existing_span.end_char > global_start and 
                        existing_span.label == span.label):
                        if span.confidence > existing_span.confidence:
                            all_spans.remove(existing_span)
                            break
                        else:
                            overlapping = True
                            break
                
                if not overlapping and span_key not in seen_spans:
                    seen_spans.add(span_key)
                    all_spans.append(PredictedSpan(
                        start_char=global_start,
                        end_char=global_end,
                        label=span.label,
                        text=text[global_start:global_end],
                        confidence=span.confidence,
                        method="sliding_window"
                    ))
            if end >= len(text):
                break
        
        all_spans.sort(key=lambda x: x.start_char)
        return all_spans
    
    def _convert_predictions_to_spans(self, predictions: np.ndarray, confidences: np.ndarray,
                                      offset_mapping: torch.Tensor, text: str, method: str) -> List[PredictedSpan]:
        """Convert token predictions to character spans using BIO tagging."""
        spans, current_span = [], None
        
        for i, (pred_id, confidence) in enumerate(zip(predictions, confidences)):
            if i >= len(offset_mapping) or (offset_mapping[i][0] == 0 and offset_mapping[i][1] == 0):
                continue
            
            label = self.id_to_label.get(pred_id, 'O')
            token_start, token_end = offset_mapping[i]
            
            if label.startswith('B-'):
                if current_span: spans.append(current_span)
                current_span = PredictedSpan(
                    int(token_start), int(token_end), label[2:], "", float(confidence), method)
            elif label.startswith('I-') and current_span:
                if label[2:] == current_span.label:
                    current_span.end_char = int(token_end)
                    current_span.confidence = min(current_span.confidence, float(confidence))
                else:
                    spans.append(current_span)
                    current_span = PredictedSpan(
                        int(token_start), int(token_end), label[2:], "", float(confidence), method)
            elif label == 'O':
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        if current_span: spans.append(current_span)
        for span in spans:
            span.text = text[span.start_char:span.end_char]
        return spans
    
    def predict_batch(self, texts: List[str]) -> List[List[PredictedSpan]]:
        """Predict spans for a batch of texts."""
        return [self.predict_text(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {'model_path': self.model_path, 'device': str(self.device),
                'label_names': self.label_names, 'num_labels': len(self.label_names),
                'context_size': self.context_size}

class TextVisualizer:
    """
    Creates a modern, interactive HTML visualization of predicted spans
    using Tailwind CSS and client-side JavaScript.
    """
    def create_html_visualization(
        self, text: str, spans: List[PredictedSpan], title: str = "Asset/Deficit Analysis"
    ) -> str:
        """
        Generates a self-contained HTML file by populating a template.
        """
        sorted_spans = sorted(spans, key=lambda x: x.start_char)
        
        # Prepare data for injection into the HTML template
        spans_json = json.dumps([asdict(s) for s in sorted_spans])
        text_json = json.dumps(text)

        # Populate the template
        html_content = HTML_REPORT_TEMPLATE.replace("%%TITLE%%", html.escape(title))
        html_content = html_content.replace("%%TEXT_JSON%%", text_json)
        html_content = html_content.replace("%%SPANS_JSON%%", spans_json)
        
        return html_content

    def _calculate_statistics(self, spans: List[PredictedSpan], text: str) -> Dict[str, Any]:
        """Calculates statistics for the analysis."""
        if not spans:
            return {'total_spans': 0, 'asset_count': 0, 'deficit_count': 0,
                    'avg_confidence': 0.0, 'text_coverage': 0.0, 'avg_span_length': 0.0}
        
        asset_count = sum(1 for s in spans if s.label == 'ASSET')
        deficit_count = sum(1 for s in spans if s.label == 'DEFICIT')
        avg_confidence = sum(s.confidence for s in spans) / len(spans) if spans else 0
        total_span_chars = sum(s.end_char - s.start_char for s in spans)
        text_coverage = (total_span_chars / len(text)) * 100 if text else 0
        avg_span_length = total_span_chars / len(spans) if spans else 0

        return {'total_spans': len(spans), 'asset_count': asset_count, 
                'deficit_count': deficit_count, 'avg_confidence': avg_confidence,
                'text_coverage': text_coverage, 'avg_span_length': avg_span_length}

