#!/bin/bash
"""
Startup script for BERT Asset/Deficit Classification Web Service
"""

echo "=== BERT Asset/Deficit Classifier Web Service ==="
echo "Starting web service..."
echo ""

# Change to the correct directory
cd /N/slate/ankichau/projects/center/bert_webapp/

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask..."
    pip install flask
fi

echo "Model location: /N/slate/ankichau/projects/center/archive/src/finetuning_bert/models/best_model"
echo ""
echo "Starting web server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the web service
python web_service.py