#!/bin/bash

# Name of your virtual environment
ENV_NAME="spatial_omics_env"

# Default notebooks directory
NOTEBOOK_DIR="notebooks"

# Activate the environment
source $ENV_NAME/bin/activate

# Check if the notebooks directory exists
if [ ! -d "$NOTEBOOK_DIR" ]; then
  echo "[!] Notebook directory '$NOTEBOOK_DIR' not found. Creating it..."
  mkdir -p "$NOTEBOOK_DIR"
fi

# Navigate to the notebooks folder
cd "$NOTEBOOK_DIR"

# Launch Jupyter Notebook
echo "[*] Launching Jupyter Notebook in $(pwd)..."
jupyter notebook