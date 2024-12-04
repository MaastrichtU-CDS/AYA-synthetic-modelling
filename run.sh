#!/bin/bash

# Run the Python script as a background task
if [ $# -eq 2 ]; then
    generator=$1
    model=$2
else
    read -p "Enter the generator (ctgan, fast_ml, or gc): " generator
    read -p "Enter the model (log_reg, rf, mlp): " model
fi

# Run the Python script with the provided arguments as a background task
python3 /root/AYA-synthetic-modelling/src/pipeline_saving.py "$generator" "$model"