#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

VENV_DIR="$HOME/.LiveWebcamAI"

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv $VENV_DIR
fi

source "$VENV_DIR/bin/activate"

pip install -r requirements.txt

python3 test.py --server
