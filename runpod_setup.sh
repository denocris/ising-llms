#!/usr/bin/env bash
set -euo pipefail

echo "Updating package index"
apt-get update

echo "Installing essentials"
apt-get install -y vim git curl

echo "Upgrading pip"
python3 -m pip install --upgrade pip

echo "Installing Python requirements"
pip install -r runpod_requirements.txt

echo "Installing Ollama"
curl -fsSL https://ollama.com/install.sh | sh

echo "Setup complete!"
