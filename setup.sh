#!/bin/bash
set -e

# Cài python3-venv nếu chưa có (cần sudo)
if ! python3 -m venv --help &>/dev/null; then
  echo "Installing python3-venv..."
  sudo apt update && sudo apt install -y python3-full python3-venv
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
npm install

echo "Done. Run: source venv/bin/activate"
