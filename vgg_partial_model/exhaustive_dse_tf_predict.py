import csv
import random
import subprocess
import time
from datetime import datetime

import numpy as np

NUMBER_OF_FUNCTION_RUNS = 20

data_blueprint = {"mean_times": None,
                  "median_times": None,
                  "max_times": None,
                  "min_times": None,
                  "mean_energies": None,
                  "median_energies": None,
                  "max_energies": None,
                  "min_energies": None,
                  "avg_script_exec_time": None,
                  "e_cores": None,
                  "p_s_cores": None,
                  "p_h_cores": None,
                  }


def init_file(time_stamp):
    with open(f"./exhaustive_dse/exhaustive_dse_results_{time_stamp}.csv", 'w', newline='') as csvfile:
        # Create a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(data_blueprint.keys())


def write_to_file(data, time_stamp):
    # Open the CSV file in write mode with a semicolon as the separator
    with open(f"./exhaustive_dse/exhaustive_dse_results_{time_stamp}.csv", 'a', newline='') as csvfile:
        # Create a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csvfile)

        # Write header
        new_line = list(data.values())
        csv_writer.writerow(new_line)


def run_eval_predict_script(affinities):
    # Define the path to your shell script
    script_path = './measure_vgg_model_energy.sh'

    # Convert the list of integers to a list of strings
    params = [str(x) for x in affinities]

    # Construct the command to run the shell script
    command = [script_path] + params

    times = []
    energies = []

    for _ in range(NUMBER_OF_FUNCTION_RUNS):

        exex_time = None
        package_energy = None

        # Run the shell script and capture the output
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            # Access the output
            script_output = result.stdout

            # print(script_output)

            # Extract values using split and convert to float
            for line in script_output.split("\n"):
                if "cpu0_package_joules" in line:
                    value = line.split("=")[-1].strip()
                    package_energy = float(value)
                if "exec_time" in line:
                    value = line.split(":")[-1].strip()
                    exex_time = float(value)

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        times += [exex_time]
        energies += [package_energy]

    return (np.mean(times), np.median(times), max(times), min(times),
            np.mean(energies), np.median(energies), max(energies), min(energies))


def refine_core_affinities(e, ps, ph):
    refined_affinities = []
    refined_affinities += [2 * p_ for p_ in range(ps)]
    refined_affinities += [p_ for p_ in range(2 * ps, 2 * ps + 2 * ph)]
    refined_affinities += [e_ for e_ in range(16, 16 + e)]

    return refined_affinities


if __name__ == '__main__':

    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    init_file(time_stamp)

    e_core_possibilities = [x for x in range(16 + 1)]
    p_s_core_possibilities = [x for x in range(8 + 1)]
    p_h_core_possibilities = [x for x in range(8 + 1)]

    st = time.time()

    counter = 0
    for e_cores in e_core_possibilities:
        for p_s_cores in p_s_core_possibilities:
            for p_h_cores in p_h_core_possibilities:

                # ignore case with 0 0 0
                if e_cores + p_s_cores + p_h_cores == 0:
                    continue

                # ignore case with ...
                if p_s_cores + p_h_cores > 8:
                    continue

                counter += 1
                core_affinities = refine_core_affinities(e_cores, p_s_cores, p_h_cores)

                st_ = time.time()
                (mean_times, median_times, max_times, min_times,
                 mean_energies, median_energies, max_energies, min_energies) = run_eval_predict_script(core_affinities)
                script_exec_time = time.time() - st_
                print(mean_times, script_exec_time / NUMBER_OF_FUNCTION_RUNS, mean_energies, e_cores, p_s_cores,
                      p_h_cores)

                write_to_file(
                    {
                        "mean_times": mean_times,
                        "median_times": median_times,
                        "max_times": max_times,
                        "min_times": min_times,
                        "mean_energies": mean_energies,
                        "median_energies": median_energies,
                        "max_energies": max_energies,
                        "min_energies": min_energies,
                        "avg_script_exec_time": script_exec_time / NUMBER_OF_FUNCTION_RUNS,
                        "e_cores": e_cores,
                        "p_s_cores": p_s_cores,
                        "p_h_cores": p_h_cores
                    }, time_stamp
                )

    print("total exploration time: ", time.time() - st)
    print("total amount of configs: ", counter)
