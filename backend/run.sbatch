#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

VENV_DIR="$HOME/.LiveWebcamAI"

git pull

ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv $VENV_DIR
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip

INSTALLED_PACKAGES=$(pip freeze | sed -e 's#=.*##' | tr '[:upper:]' '[:lower:]')

install_missing_packages() {
        local changed=0
        while IFS= read -r line || [ -n "$line" ]; do
                original_line=$line
                line=$(echo "$line" | tr '[:upper:]' '[:lower:]')
                [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

                if ! echo "$INSTALLED_PACKAGES" | grep -Fxq "$line"; then
                        echo "Installing missing or mismatched package: $line"
                        pip install "$original_line"
                        changed=1
                fi
        done < requirements.txt

        if [[ $changed -eq 1 ]]; then
                INSTALLED_PACKAGES=$(pip freeze | sed -e 's#=.*##' | tr '[:upper:]' '[:lower:]')
        fi
}

install_missing_packages

python3 main.py --server $*
