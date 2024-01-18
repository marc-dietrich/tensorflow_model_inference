#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <integer_parameter>"
    exit 1
fi

# Start cpu-energy-meter in the background
cpu-energy-meter &
PID=$!

# Run the ./app with the passed integer parameter
python ./main.py -s 23 -c 16 -i "$1"

# Send SIGINT signal to cpu-energy-meter
kill -SIGINT $PID
