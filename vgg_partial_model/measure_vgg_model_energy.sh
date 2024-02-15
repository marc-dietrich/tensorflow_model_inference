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
output=$(python eval_saved_model.py --core_affinities $CORE_AFFINITIES)


captured_line_exec_time=$(echo "$output" | grep "exec_time")
echo "$captured_line_exec_time"

# Send SIGINT signal to cpu-energy-meter
kill -SIGINT $PID
