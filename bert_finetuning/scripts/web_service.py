#!/usr/bin/env python3
"""
Web Service for BERT Asset/Deficit Classification.

A Flask web application that provides an interactive interface for
the trained BERT model to classify text for asset/deficit patterns.

Usage:
    python scripts/web_service.py --model models/best_model --port 5000
"""

from flask import Flask, request, render_template, jsonify, send_from_directory
import json
import os
import logging
from pathlib import Path
import sys
import traceback
from datetime import datetime
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.inference import AssetDeficitInference, TextVisualizer
from config.helpers import setup_logging

# Initialize Flask app
app = Flask(__name__, template_folder=str(project_root / 'templates'))
app.config['SECRET_KEY'] = 'bert-finetuning-secret-key-change-in-production'

# Global variables for model
model_instance = None
visualizer = None
logger = None

@app.route('/')
def index():
    """Main page with text input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return asset/deficit spans."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Check if model is loaded
        if model_instance is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Predict spans
        confidence_threshold = data.get('confidence_threshold', 0.5)
        spans = model_instance.predict_text(text, confidence_threshold)
        
        # Convert spans to dict format
        spans_data = [
            {
                'start_char': span.start_char,
                'end_char': span.end_char,
                'label': span.label,
                'text': span.text,
                'confidence': span.confidence,
                'token_scores': span.token_scores
            }
            for span in spans
        ]
        
        # Generate statistics
        stats = {
            'total_spans': len(spans),
            'asset_spans': len([s for s in spans if s.label == 'ASSET']),
            'deficit_spans': len([s for s in spans if s.label == 'DEFICIT']),
            'avg_confidence': sum(s.confidence for s in spans) / len(spans) if spans else 0
        }
        
        response_data = {
            'spans': spans_data,
            'stats': stats,
            'text_length': len(text),
            'processed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Analyzed text ({len(text)} chars): {len(spans)} spans found")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def create_visualization():
    """Create HTML visualization for analyzed text."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        spans_data = data.get('spans', [])
        title = data.get('title', 'BERT Asset/Deficit Analysis')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Convert spans data back to PredictedSpan objects
        from config.inference import PredictedSpan
        spans = [
            PredictedSpan(
                start_char=span_data['start_char'],
                end_char=span_data['end_char'],
                label=span_data['label'],
                text=span_data['text'],
                confidence=span_data['confidence'],
                token_scores=span_data.get('token_scores', [])
            )
            for span_data in spans_data
        ]
        
        # Generate HTML visualization
        html_content = visualizer.create_html_visualization(text, spans, title)
        
        logger.info(f"Generated visualization for {len(spans)} spans")
        
        return jsonify({
            'html': html_content,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500

@app.route('/model/info')
def model_info():
    """Get information about the loaded model."""
    if model_instance is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_path': model_instance.model_path,
            'device': str(model_instance.device),
            'label_names': model_instance.label_names,
            'max_length': model_instance.max_length,
            'config': model_instance.config
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'model_loaded': model_instance is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def load_model(model_path: str, device: str = 'auto'):
    """Load the BERT model for inference."""
    global model_instance, visualizer
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_instance = AssetDeficitInference(model_path, device)
        visualizer = TextVisualizer()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def create_template_if_not_exists():
    """Create basic HTML template if it doesn't exist."""
    template_dir = project_root / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    template_file = template_dir / 'index.html'
    
    if not template_file.exists():
        template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERT Asset/Deficit Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .span-asset {
            background-color: #dcfce7;
            border: 1px solid #16a34a;
            border-radius: 4px;
            padding: 2px 4px;
            margin: 0 1px;
        }
        .span-deficit {
            background-color: #fef2f2;
            border: 1px solid #dc2626;
            border-radius: 4px;
            padding: 2px 4px;
            margin: 0 1px;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto p-6 max-w-4xl">
        <h1 class="text-3xl font-bold text-gray-900 mb-8">BERT Asset/Deficit Classifier</h1>
        
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Analyze Text</h2>
            <textarea id="inputText" 
                      class="w-full h-40 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" 
                      placeholder="Enter your text here for asset/deficit analysis..."></textarea>
            
            <div class="flex items-center space-x-4 mt-4">
                <div class="flex items-center">
                    <label class="text-sm text-gray-600 mr-2">Confidence Threshold:</label>
                    <input type="range" id="confidenceThreshold" min="0" max="1" step="0.1" value="0.5" class="mr-2">
                    <span id="confidenceValue">0.5</span>
                </div>
                <button id="analyzeBtn" 
                        class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors">
                    Analyze Text
                </button>
            </div>
        </div>
        
        <div id="results" class="hidden bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Analysis Results</h2>
            <div id="stats" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6"></div>
            <div id="analyzedText" class="text-lg leading-relaxed border border-gray-200 rounded-lg p-4 bg-gray-50"></div>
        </div>
        
        <div id="spans" class="hidden bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold mb-4">Detected Spans</h2>
            <div id="spansList" class="space-y-3"></div>
        </div>
        
        <div id="loading" class="hidden text-center py-8">
            <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <p class="mt-2 text-gray-600">Analyzing text...</p>
        </div>
    </div>

    <script>
        const analyzeBtn = document.getElementById('analyzeBtn');
        const inputText = document.getElementById('inputText');
        const confidenceThreshold = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');
        const results = document.getElementById('results');
        const spans = document.getElementById('spans');
        const loading = document.getElementById('loading');

        // Update confidence value display
        confidenceThreshold.addEventListener('input', (e) => {
            confidenceValue.textContent = e.target.value;
        });

        // Analyze button click
        analyzeBtn.addEventListener('click', async () => {
            const text = inputText.value.trim();
            if (!text) {
                alert('Please enter some text to analyze.');
                return;
            }

            // Show loading
            loading.classList.remove('hidden');
            results.classList.add('hidden');
            spans.classList.add('hidden');
            analyzeBtn.disabled = true;

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        confidence_threshold: parseFloat(confidenceThreshold.value)
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    displayResults(text, data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error analyzing text: ' + error.message);
            } finally {
                loading.classList.add('hidden');
                analyzeBtn.disabled = false;
            }
        });

        function displayResults(text, data) {
            // Display statistics
            const statsHtml = `
                <div class="bg-blue-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-blue-600">${data.stats.total_spans}</div>
                    <div class="text-sm text-gray-600">Total Spans</div>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-green-600">${data.stats.asset_spans}</div>
                    <div class="text-sm text-gray-600">Asset Spans</div>
                </div>
                <div class="bg-red-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-red-600">${data.stats.deficit_spans}</div>
                    <div class="text-sm text-gray-600">Deficit Spans</div>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-purple-600">${data.stats.avg_confidence.toFixed(2)}</div>
                    <div class="text-sm text-gray-600">Avg Confidence</div>
                </div>
            `;
            document.getElementById('stats').innerHTML = statsHtml;

            // Display highlighted text
            let highlightedText = text;
            const sortedSpans = data.spans.sort((a, b) => b.start_char - a.start_char);
            
            sortedSpans.forEach(span => {
                const spanClass = span.label === 'ASSET' ? 'span-asset' : 'span-deficit';
                const beforeText = highlightedText.substring(0, span.start_char);
                const spanText = highlightedText.substring(span.start_char, span.end_char);
                const afterText = highlightedText.substring(span.end_char);
                
                const wrappedSpan = `<span class="${spanClass}" title="Confidence: ${span.confidence.toFixed(3)}">${spanText}</span>`;
                highlightedText = beforeText + wrappedSpan + afterText;
            });

            document.getElementById('analyzedText').innerHTML = highlightedText;

            // Display span details
            const spansHtml = data.spans.map(span => {
                const labelColor = span.label === 'ASSET' ? 'text-green-600' : 'text-red-600';
                return `
                    <div class="border border-gray-200 rounded-lg p-4">
                        <div class="flex justify-between items-start mb-2">
                            <span class="text-lg font-semibold ${labelColor}">${span.label}</span>
                            <span class="text-sm text-gray-500">Confidence: ${span.confidence.toFixed(3)}</span>
                        </div>
                        <div class="text-gray-700 mb-2">"${span.text}"</div>
                        <div class="text-sm text-gray-500">
                            Position: ${span.start_char}-${span.end_char} | Length: ${span.text.length} chars
                        </div>
                    </div>
                `;
            }).join('');

            document.getElementById('spansList').innerHTML = spansHtml || '<p class="text-gray-500 text-center">No spans detected.</p>';

            // Show results
            results.classList.remove('hidden');
            spans.classList.remove('hidden');
        }
    </script>
</body>
</html>
        '''
        
        with open(template_file, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Created template file: {template_file}")

def main():
    """Main function to run the web service."""
    parser = argparse.ArgumentParser(description="BERT Asset/Deficit Classification Web Service")
    parser.add_argument('--model', required=True, help='Path to trained model directory')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Create template if needed
        create_template_if_not_exists()
        
        # Load model
        load_model(args.model, args.device)
        
        # Run Flask app
        logger.info(f"Starting web service on {args.host}:{args.port}")
        logger.info(f"Model loaded from: {args.model}")
        logger.info(f"Access the interface at: http://{args.host}:{args.port}")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Web service stopped by user")
    except Exception as e:
        logger.error(f"Failed to start web service: {str(e)}")
        raise

if __name__ == "__main__":
    main()