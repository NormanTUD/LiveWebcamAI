#!/bin/bash

VENV_DIR="$HOME/.LiveWebcamAI"

if [[ ! -d $VENV_DIR ]]; then
	python3 -mvenv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip

INSTALLED_PACKAGES=$(pip freeze | sed -e 's#=.*##')

install_missing_packages() {
	local changed=0
	while IFS= read -r line || [ -n "$line" ]; do
		echo "line: $line"
		[[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

		if ! echo "$INSTALLED_PACKAGES" | grep -Fxq "$line"; then
			echo "Installing missing or mismatched package: $line"
			pip install "$line"
			changed=1
		fi
	done < requirements.txt

	if [[ $changed -eq 1 ]]; then
		INSTALLED_PACKAGES=$(pip freeze | sed -e 's#=.*##')
	fi
}

install_missing_packages

python3 main.py "$@"
