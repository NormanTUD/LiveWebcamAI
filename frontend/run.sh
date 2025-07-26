#!/bin/bash

VENV_DIR="$HOME/.LiveWebcamAI"

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv $VENV_DIR
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip

pip install -r requirements.txt

python3 main.py
