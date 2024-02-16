import csv
import subprocess
import time
from datetime import datetime


def init_file(time_stamp):
    with open(f"./exhaustive_dse/exhaustive_dse_results_{time_stamp}.csv", 'w', newline='') as csvfile:
        # Create a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csvfile)

        # Write header
        header = ['exec_time', 'script_exec_time', 'package_energy', 'e_cores', 'st_cores', 'ht_cores']
        csv_writer.writerow(header)


def write_to_file(exec_time, script_exec_time, package_energy, e_cores, p_s_cores, p_h_cores, time_stamp):
    # Open the CSV file in write mode with a semicolon as the separator
    with open(f"./exhaustive_dse/exhaustive_dse_results_{time_stamp}.csv", 'a', newline='') as csvfile:
        # Create a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csvfile)

        # Write header
        new_line = [exec_time, script_exec_time, package_energy, e_cores, p_s_cores, p_h_cores]
        csv_writer.writerow(new_line)


def run_eval_predict_script(affinities):

    # Define the path to your shell script
    script_path = './measure_vgg_model_energy.sh'

    # Convert the list of integers to a list of strings
    params = [str(x) for x in affinities]

    # Construct the command to run the shell script
    command = [script_path] + params

    exex_time = None
    package_energy = None

    # Run the shell script and capture the output
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Access the output
        script_output = result.stdout

        #print(script_output)

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

    return exex_time, package_energy


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
                exec_time, package_energy = run_eval_predict_script(core_affinities)
                script_exec_time = time.time() - st_
                print(exec_time, script_exec_time, package_energy, e_cores, p_s_cores, p_h_cores)

                write_to_file(exec_time, script_exec_time, package_energy, e_cores, p_s_cores, p_h_cores, time_stamp)

    print("total exploration time: ", time.time() - st)
    print("total amount of configs: ", counter)
