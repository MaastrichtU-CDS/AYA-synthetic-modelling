#!/bin/bash

# Default values for optional usage
architecture=""
algorithm=""
test_df=""
train_df=""

# Parse flagged arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --architecture) architecture="$2"; shift ;;
        --algorithm) algorithm="$2"; shift ;;
        --test_df) test_df="$2"; shift ;;
        --train_df) train_df="$2"; shift ;;
        --) shift; break ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$architecture" ] || [ -z "$algorithm" ] || [ -z "$test_df" ]; then
    read -p "Enter the architecture (ctgan, fast_ml, gc, or dp_cgan): " architecture
    read -p "Enter the algorithm (log_reg, rf, mlp): " algorithm
    read -p "Enter the test data: " test_df
    read -p "Enter the train data: " train_df
fi

# Validate the inputs
if [[ ! "$architecture" =~ ^(ctgan|fast_ml|gc|dp_cgan)$ ]]; then
    echo "Invalid architecture: $architecture. Choose from ctgan, fast_ml, gc, or dp_cgan."
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

if [ -z "$train_df" ]; then
    echo "Train df is not provided because it's in pipeline loop"
else
    echo "Train df is provided. Hence, not running from pipeline loop."
fi

# Run the Python script with flagged arguments
python /root/AYA-synthetic-modelling/src/pipeline_saving.py --architecture "$architecture" --algorithm "$algorithm" --test_df "$test_df"

# Move the nohup.out file to the root directory & rename it if it exists
if [ -f nohup.out ]; then
    if mv nohup.out "/root/${architecture}_${algorithm}.out"; then
        echo "Moved nohup.out to /root/${architecture}_${algorithm}.out"
    else
        echo "Failed to move nohup.out"
        exit 1
    fi
fi
