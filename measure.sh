#!/bin/bash

# Start cpu-energy-meter in the background
cpu-energy-meter &
PID=$!

# Run the ./app with the passed integer parameter
# Run the ./app with the passed integer parameters
output=$(python ./main.py -i 0 -m VGG -ind "$@")

# Capture the desired line from the output
captured_line_exec_time = $(echo "$output" | grep "exec.time")
captured_line_latency = $(echo "$output" | grep "latency")
captured_line_accuracy = $(echo "$output" | grep "accuracy")

# Print the captured line
echo "$output"
echo "$captured_line_exec_time"
echo "$captured_line_latency"
echo "$captured_line_accuracy"

# Send SIGINT signal to cpu-energy-meter
kill -SIGINT $PID
