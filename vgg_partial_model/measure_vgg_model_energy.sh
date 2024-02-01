#!/bin/bash

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_cores)
            NUM_CORES="$2"
            shift 2
            ;;
        --core_affinities)
            CORE_AFFINITIES=($2)
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$NUM_CORES" ] || [ -z "$CORE_AFFINITIES" ]; then
    echo "Usage: $0 --num_cores <num_cores> --core_affinities <core_affinities>"
    exit 1
fi

cpu-energy-meter -r &
PID=$!

# Call the Python script with parameters
python your_script.py --num_cores "$NUM_CORES" --core_affinities "${CORE_AFFINITIES[@]}"

# Send SIGINT signal to cpu-energy-meter
kill -SIGINT $PID
