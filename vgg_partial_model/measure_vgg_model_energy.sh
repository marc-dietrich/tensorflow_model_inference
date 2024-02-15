#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <core_affinities>"
    exit 1
fi

# Extract core affinities from arguments
CORE_AFFINITIES=$@

cpu-energy-meter -r &
PID=$!

# Call the Python script with parameters
python eval_saved_model.py --core_affinities "$CORE_AFFINITIES"

# Send SIGINT signal to cpu-energy-meter
kill -SIGINT $PID
