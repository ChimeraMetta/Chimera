#!/bin/bash
#
# PostgreSQL Query Optimizer Demo - Setup Script
#
# This script sets up everything needed to run the MeTTa-powered
# query optimizer demo on a fresh Mac.
#
# Usage:
#   chmod +x setup_demo.sh
#   ./setup_demo.sh
#

set -e

echo "========================================"
echo "PostgreSQL Query Optimizer - Setup"
echo "========================================"
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add brew to PATH for Apple Silicon Macs
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "Homebrew found."
fi

# Check for Python 3.12
if ! brew list python@3.12 &> /dev/null; then
    echo "Installing Python 3.12..."
    brew install python@3.12
else
    echo "Python 3.12 found."
fi

# Determine Python 3.12 path
if [[ -f /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
    PYTHON_BIN="/opt/homebrew/opt/python@3.12/bin/python3.12"
elif [[ -f /usr/local/opt/python@3.12/bin/python3.12 ]]; then
    PYTHON_BIN="/usr/local/opt/python@3.12/bin/python3.12"
else
    echo "Error: Could not find Python 3.12 binary"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Create virtual environment
VENV_DIR=".venv312"
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv $VENV_DIR
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install hyperon --quiet
pip install fastapi uvicorn jinja2 requests tqdm psutil GitPython inquirer python-multipart colorama blessed --quiet

# Verify hyperon installation
echo ""
echo "Verifying Hyperon installation..."
python -c "from hyperon import MeTTa; m = MeTTa(); print('Hyperon', __import__('hyperon').__version__, 'installed successfully!')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run the demo:"
echo ""
echo "  source .venv312/bin/activate"
echo "  python pg_optimizer_server.py"
echo ""
echo "Then open http://localhost:8001 in your browser."
echo ""
echo "========================================"
