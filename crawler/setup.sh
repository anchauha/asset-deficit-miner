#!/bin/bash

# URL Content Extractor Setup Script
# This script sets up the environment for the standalone URL content extractor

echo "🔧 Setting up URL Content Extractor..."
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

# Make the script executable
chmod +x url_extractor.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Quick Start:"
echo "  1. Place your CSV file with URLs in data/input/"
echo "  2. Run: python3 url_extractor.py --csv_file data/input/your_file.csv --url_column your_url_column"
echo "  3. Check results in data/output/"
echo ""
echo "📖 For more options, run: python3 url_extractor.py --help"
echo ""