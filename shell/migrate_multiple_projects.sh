#!/bin/bash

# Define an array with items containing spaces
project_names=(
    "nik-dbrx-quant-eval"
)

# Loop over the array elements
for name in "${project_names[@]}"; do
    w2m --wandb_project_name=$name
done
