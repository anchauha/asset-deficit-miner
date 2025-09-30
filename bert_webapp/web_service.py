#!/usr/bin/env python3
"""
Web Service for BERT Asset/Deficit Classification.

A Flask web application that provides an interactive interface for
the trained BERT model to classify text for asset/deficit patterns.
"""

from flask import Flask, request, render_template, jsonify, send_from_directory
import json
import os
import logging
from pathlib import Path
import sys
import traceback
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference import AssetDeficitInference, TextVisualizer
from utils import setup_logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Global variables for model
model_instance = None
visualizer = None

# Configuration
MODEL_PATH = "/N/slate/ankichau/projects/center/bert_webapp/models/best_model"
CONTEXT_SIZE = 512

def initialize_model():
    """Initialize the BERT model and visualizer."""
    global model_instance, visualizer
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model_instance = AssetDeficitInference(MODEL_PATH, context_size=CONTEXT_SIZE)
        visualizer = TextVisualizer()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main interface page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text using the BERT model."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Check if model is loaded
        if model_instance is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Add debugging information
        text_length = len(text)
        print(f"DEBUG: Analyzing text of length {text_length} characters")
        
        # Run inference
        spans = model_instance.predict_text(text)
        
        print(f"DEBUG: Found {len(spans)} spans")
        for i, span in enumerate(spans[:5]):  # Log first 5 spans
            print(f"DEBUG: Span {i}: {span.label} '{span.text[:50]}...' (conf: {span.confidence:.3f})")
        
        # Convert spans to JSON-serializable format
        results = []
        for span in spans:
            results.append({
                'start_char': span.start_char,
                'end_char': span.end_char,
                'label': span.label,
                'text': span.text,
                'confidence': round(span.confidence, 3),
                'method': getattr(span, 'method', 'inference')
            })
        
        # Generate HTML visualization
        html_content = visualizer.create_html_visualization(
            text, spans, title="Asset/Deficit Analysis - Web Interface"
        )
        
        # Return results
        return jsonify({
            'success': True,
            'text': text,
            'spans': results,
            'html_visualization': html_content,
            'summary': {
                'total_spans': len(results),
                'asset_spans': len([s for s in results if s['label'] == 'ASSET']),
                'deficit_spans': len([s for s in results if s['label'] == 'DEFICIT']),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"Error in analyze_text: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'model_loaded': model_instance is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/model-info')
def model_info():
    """Get information about the loaded model."""
    if model_instance is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_path': MODEL_PATH,
            'context_size': CONTEXT_SIZE,
            'model_type': 'BERT for Token Classification',
            'labels': ['ASSET', 'DEFICIT'],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

if __name__ == '__main__':
    # Setup logging
    setup_logging('INFO')
    
    print("="*60)
    print("BERT Asset/Deficit Classification Web Service")
    print("="*60)
    
    # Initialize model
    if not initialize_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    print(f"Model loaded from: {MODEL_PATH}")
    print("Starting web server...")
    print("="*60)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,
        debug=True,
        threaded=True
    )