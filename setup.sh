#!/bin/bash

# SmartTicket AI Setup Script

echo "================================="
echo "SmartTicket AI Setup"
echo "================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "Error: Python 3.9 or higher is required"
    exit 1
fi
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models logs evaluation_results
echo "✓ Directories created"
echo ""

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Build vector store
echo "Building vector store..."
python vector_store/build_index.py
echo "✓ Vector store built"
echo ""

# Train models
echo "Training classification models..."
python training/train.py
echo "✓ Models trained"
echo ""

echo "================================="
echo "Setup Complete!"
echo "================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI or Anthropic API key"
echo "2. Run the API: uvicorn api.main:app --reload"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "To run tests: pytest tests/"
echo "To evaluate models: python training/evaluate.py"
echo ""
