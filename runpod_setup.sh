#!/usr/bin/env bash
set -euo pipefail

echo "Updating package index"
apt-get update

echo "Installing essentials"
apt-get install -y vim git curl

echo "Upgrading pip"
python3 -m pip install --upgrade pip

echo "Installing Python requirements"
pip3 install -r requirements.txt
pip3 install matplotlib openai groq tqdm transformers ollama

echo "Installing Ollama"
curl -fsSL https://ollama.com/install.sh | sh

nohup ollama serve > ollama.log 2>&1 &

ollama pull gemma3:1b

echo "Setup complete!"
