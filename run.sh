#!/bin/bash

# Default values for optional usage
architecture=""
algorithm=""
test_df=""

# Parse flagged arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --architecture) architecture="$2"; shift ;;
        --algorithm) algorithm="$2"; shift ;;
        --test_df) test_df="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Interactive mode if no flags are passed
if [ -z "$architecture" ] || [ -z "$algorithm" ]; then
    read -p "Enter the architecture (ctgan, fast_ml, or gc): " architecture
    read -p "Enter the algorithm (log_reg, rf, mlp): " algorithm
    read -p "Enter the test data: " test_df
fi

# Validate the inputs
if [[ ! "$architecture" =~ ^(ctgan|fast_ml|gc)$ ]]; then
    echo "Invalid architecture: $architecture. Choose from ctgan, fast_ml, or gc."
    exit 1
fi

if [[ ! "$algorithm" =~ ^(log_reg|rf|mlp)$ ]]; then
    echo "Invalid algorithm: $algorithm. Choose from log_reg, rf, or mlp."
    exit 1
fi

if [ -z "$test_df" ]; then
    echo "Test df is required."
    exit 1
fi

# Run the Python script with flagged arguments
python3 /root/AYA-synthetic-modelling/src/pipeline_saving.py --architecture "$architecture" --algorithm "$algorithm" --test_df "$test_df"