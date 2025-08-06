#!/usr/bin/env bash
set -euo pipefail

# 1) Update the package index
sudo apt-get update

# 2) Install essentials
sudo apt-get install -y vim git curl

# 3) Upgrade pip
python3 -m pip install --upgrade pip

# 4) Install Python requirements
pip install -r runpod_requirements.txt

# 5) Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

echo "✅ Setup complete!"
