# Environment variables
PYTHON = python
PIP = pip
REMOVE = rm -rf
CREATE = mkdir -p
PRINT = @echo

# Paths
LOG_PATH = ./logs

# targets
## setup :		Setup the virtual environment.
.PHONY: setup

## install :		Install dependencies.
.PHONY: install

## update :		Update dependencies.
.PHONY: update

## experiments : 	Run experiments.
.PHONY: experiments

## clean :		Clean up.
.PHONY: clean

## visualize :		Visualize results on tensorboard.
.PHONY: visualize

## reset :		Reset the environment.
.PHONY: reset


# recipes
setup: create_env create_temp
install: install_requirements
update: update_requirements

# rules
create_env:
	$(PRINT) "Creating Virtual Environment..."
	$(PYTHON) -m venv ./.venv
	$(PRINT)

install_requirements:
	$(PRINT) "Installing Dependencies..."
	$(PIP) install -r ./requirements.txt
	$(PRINT) ""

update_requirements:
	$(PRINT) "Updating Dependencies..."
	$(PIP) freeze > ./requirements.txt
	$(PRINT) ""