import argparse
import math
import multiprocessing
import time

import keras
import numpy as np
import psutil
import tensorflow as tf
from tqdm import tqdm


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    x_train = tf.repeat(x_train, 3, axis=3)
    x_test = tf.repeat(x_test, 3, axis=3)
    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]
    print(np.shape(x_test[0]))
    # x_test = list(map(lambda x: np.reshape(x, newshape=(1, 32, 32, 3)), x_test))
    x_test = list(map(lambda x: np.expand_dims(x, axis=0), x_test))
    print(np.shape(x_test[0]))
    return x_test  # , y_val


def get_interpreters(partial_model_dir_path, model_name, indices):
    interpreters = []
    for i in indices:
        interpreter = tf.lite.Interpreter(partial_model_dir_path + "/" + model_name + "_" + str(i) + ".tflite")
        interpreter.allocate_tensors()
        interpreters.append(interpreter)
    return interpreters


interpreters = get_interpreters("./vgg_partial_model/vgg_conv_base_model", "vgg_conv_base", [i for i in range(19)])
interpreters += get_interpreters("./vgg_partial_model/vgg_extension_model", "vgg_model", [i for i in range(4)])

'''
for interpreter in interpreters:
    print(interpreter.get_input_details())
    print(interpreter.get_output_details())
    print("")
'''


def run_stage(stage_idx, input):
    interpreter = interpreters[stage_idx]
    # print(np.shape(input))
    # print(interpreters.index(interpreter))
    # print(interpreter.get_input_details())

    if stage_idx == 19:
        input = np.reshape(input, newshape=(1, 512))

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


# Function to simulate a pipeline stage
def assignment_worker(in_queue, out_queue, assignment, core_id):
    # Set CPU affinity for this process
    st = time.time()
    psutil.Process().cpu_affinity([core_id])
    print("aff. time: ", time.time() - st)

    while True:
        data = in_queue.get()
        if data is None:
            out_queue.put(None)
            break  # Exit loop when done
        # Perform some processing on data
        for stage in assignment:
            '''
            # TODO: make use of stage..
            time.sleep(0.001)  # -> simulate work which needs more effort...
            # otherwise ... using single thread is faster than piepline parallelism (probably due to memory overhead)
            data = data * 2  # Example processing
            '''
            data = run_stage(stage, data)
        out_queue.put(data)


def main(num_stages, assignments):
    # Create queues for communication between stages
    queues = [multiprocessing.Queue() for _ in range(len(assignments) + 1)]

    # Create processes for each stage
    processes = [
        multiprocessing.Process(target=assignment_worker, args=(queues[idx], queues[idx + 1], assignment, core_id))
        for idx, (core_id, assignment) in enumerate(assignments.items())
    ]

    # Start processes
    for process in processes:
        process.start()

    # Input data to the first stage
    data = get_data()
    for item in data:
        queues[0].put(item)

    # Signal the end of input by sending None through the queue
    queues[0].put(None)

    # Collect results from the last stage
    results = []
    with tqdm(total=len(data)) as bar:
        while True:
            result = queues[-1].get()
            if result is None:
                break
            results.append(result)
            bar.update()
        bar.close()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    return results


def get_assignments(num_stages, cores):
    num_cores = len(cores)
    assignments = {}
    counter = []
    stages_per_cores = num_stages / num_cores
    floor_ = math.floor(stages_per_cores)
    for _ in range(num_cores):
        counter.append(floor_)
    rest = num_stages - (floor_ * num_cores)
    for core_idx in range(rest):
        counter[core_idx] += 1

    start = 0
    for count, core_id in zip(counter, cores):
        assignments[core_id] = list(range(start, start + count))
        start += count

    print(assignments)

    return assignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Parallelism")
    parser.add_argument("-s", "--stages", type=int, help="Number of stages in the pipeline", required=True)
    parser.add_argument("-c", "--cores", nargs='+', type=int, help="List of cores for each stage", required=True)
    args = parser.parse_args()

    num_stages = args.stages
    cores = args.cores

    if len(cores) > num_stages:
        raise AttributeError("Too many cores passed .. do not provide more cores than stages!")

    if len(cores) != num_stages:
        print("Amount of cores and amount of stages differ --> assigning multiple stages/threads to on core!")

    assignments = get_assignments(num_stages, cores)
    '''
    assignments = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12],
        [13, 14],
        [15],
        [16],
        [17, 18],
        [19, 20, 21, 22]
    ]
    '''

    st = time.time()
    result = main(num_stages, assignments)
    print("Exec. time: ", time.time() - st)
    # print("Result:", result[:10])
