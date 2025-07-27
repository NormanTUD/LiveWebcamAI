#!/bin/bash

VENV_DIR="$HOME/.LiveWebcamAI"

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install -q --upgrade pip

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

python3 main.py "$@"
