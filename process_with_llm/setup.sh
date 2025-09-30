#!/bin/bash

# LLM Processing Feature Setup Script
# This script sets up the environment for LLM-based asset vs deficit language analysis

echo "🧠 Setting up LLM Processing Feature..."
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install dependencies
echo "📦 Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
else
    echo "❌ Error installing dependencies. Please check your internet connection and try again."
    exit 1
fi

# Create sample data if it doesn't exist
if [ ! -f "data/input/sample_data.json" ]; then
    echo "📄 Sample data will be created automatically when scripts are first run."
fi

# Make scripts executable
chmod +x scripts/*.py

# Check for GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()) if torch.cuda.is_available() else None"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Available Scripts:"
echo "  • process_with_ollama.py     - Use local Ollama models"
echo "  • process_with_huggingface.py - Use HuggingFace models"
echo ""
echo "📖 Quick Start:"
echo "  1. For Ollama: Start ollama service (ollama serve)"
echo "  2. Place your JSON data in data/input/"
echo "  3. Run: python3 scripts/process_with_ollama.py --input data/input/your_file.json"
echo "  4. Or: python3 scripts/process_with_huggingface.py --input data/input/your_file.json"
echo "  5. Check results in data/output/"
echo ""
echo "💡 For more options, run: python3 scripts/[script_name].py --help"
echo ""