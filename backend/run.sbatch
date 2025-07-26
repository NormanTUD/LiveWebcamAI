#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

VENV_DIR="$HOME/.LiveWebcamAI"

ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv $VENV_DIR
fi

source "$VENV_DIR/bin/activate"

pip install -r requirements.txt

python3 test.py --server
