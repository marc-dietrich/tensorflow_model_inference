import argparse
import math
import multiprocessing
import os
import time

import keras
import numpy as np
import psutil
import tensorflow as tf
from tqdm import tqdm


def get_data():
    x_train = np.load("./mnist_dataset/train.npz")["arr_0"]
    y_train = np.load("./mnist_dataset/train.npz")["arr_1"]

    x_test = np.load("./mnist_dataset/test.npz")["arr_0"]
    y_test = np.load("./mnist_dataset/test.npz")["arr_1"]

    x_val = np.load("./mnist_dataset/val.npz")["arr_0"]
    y_val = np.load("./mnist_dataset/val.npz")["arr_1"]

    # print(x_train, y_train)
    # return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    # x_test = np.array(list(map(lambda x: np.expand_dims(x, axis=0), x_test)))

    # return x_test[:1000], y_test[:1000]

    x_test = [x[None, :, :, :] for x in x_test]

    return x_test, y_test


def get_interpreters(partial_model_dir_path, model_name, size):
    interpreters = []
    for i in range(size):
        interpreter = tf.lite.Interpreter(partial_model_dir_path + "/" + model_name + "_" + str(i) + ".tflite")
        interpreter.allocate_tensors()
        interpreters.append(interpreter)
    return interpreters


interpreters = get_interpreters(partial_model_dir_path="vgg_partial_model/vgg_conv_base_model",
                                model_name="conv_base_model",
                                size=19
                                )
interpreters += get_interpreters(partial_model_dir_path="vgg_partial_model/vgg_extension_model",
                                 model_name="extension_model",
                                 size=4
                                 )

# layers = conv_base_model.layers


'''
for interpreter in interpreters:
    print(interpreter.get_input_details())
    print(interpreter.get_output_details())
    print("")
'''

conv_base_model: keras.models.Sequential = keras.models.load_model(
    "vgg_partial_model/vgg_conv_base_model/conv_base_model.keras")
extension_model: keras.models.Sequential = keras.models.load_model(
    "./vgg_partial_model/vgg_extension_model/extension_model.keras")
layers = conv_base_model.layers
layers += extension_model.layers


def run_stage(stage_idx, input, input_idx, type_):
    if type_ == "interpreter":
        interpreter = interpreters[stage_idx]

    # print(np.shape(input))
    # print(interpreters.index(interpreter))
    # print(interpreter.get_input_details())

    st = time.time()
    if stage_idx == 19:
        input = np.reshape(input, newshape=(1, 512))

    if type_ == "interpreter":
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    elif type_ == "layers":
        output = layers[stage_idx](input, training=False)
    else:
        raise RuntimeError("only type 'interpreter' or 'layers' possible")

    et = time.time() - st
    return output, et


# Function to simulate a pipeline stage
def assignment_worker(in_queue, out_queue, assignment, core_id, latencies, type_):
    # Set CPU affinity for this process
    st = time.time()
    psutil.Process().cpu_affinity([core_id])
    #print("aff. time: ", time.time() - st)

    counter = 0
    ets = [[0 for _ in range(10_000)] for _ in assignment]
    while True:
        data = in_queue.get()
        if data is None:
            out_queue.put(None)
            break  # Exit loop when done
        # Perform some processing on data
        for nr, stage in enumerate(assignment):
            '''
            # TODO: make use of stage..
            time.sleep(0.001)  # -> simulate work which needs more effort...
            # otherwise ... using single thread is faster than piepline parallelism (probably due to memory overhead)
            data = data * 2  # Example processing
            '''
            data, et = run_stage(stage, data, counter, type_)
            ets[nr][counter] = et
            counter += 1
        counter = 0
        out_queue.put(data)
        for et, a in zip(ets, assignment):
            latencies[a] = np.sum(et)


def main(num_stages, assignments, data, type_):
    # Create queues for communication between stages
    queues = [multiprocessing.Queue() for _ in range(len(assignments) + 1)]
    latencies = multiprocessing.Array('d', 23)

    # Create processes for each stage
    processes = [
        multiprocessing.Process(target=assignment_worker,
                                args=(queues[idx], queues[idx + 1], assignment, core_id, latencies, type_))
        for idx, (core_id, assignment) in enumerate(assignments.items())
    ]

    # Start processes
    st = time.time()
    for process in processes:
        process.start()

    # Input data to the first stage
    x_test = data[0]
    for item in x_test:
        queues[0].put(item)

    # Signal the end of input by sending None through the queue
    queues[0].put(None)

    # Collect results from the last stage
    results = []
    with tqdm(total=len(x_test)) as bar:
        while True:
            result = queues[-1].get()
            if result is None:
                break
            results.append(result)
            bar.update()
        bar.close()
        et = time.time() - st

    # Wait for all processes to finish
    for process in processes:
        process.join()

    acc_counter = 0
    y_test = data[1]
    for result, y in zip(results, y_test):
        # print(result, y)
        if np.argmax(result) == y:
            acc_counter += 1

    avg_latency = np.mean(latencies)

    return results, acc_counter / len(y_test), et, avg_latency


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


def run_two_split_model(data, core_aff, type_):
    psutil.Process().cpu_affinity([core_aff])

    x_test = data[0]
    y_test = data[1]

    # init interpreter
    if type_ == "tflite":
        conv_base_interpreter = tf.lite.Interpreter(
            "vgg_partial_model/vgg_conv_base_model/conv_base_model.tflite"
        )
        conv_base_interpreter.allocate_tensors()
        extension_interpreter = tf.lite.Interpreter(
            "vgg_partial_model/vgg_extension_model/extension_model.tflite"
        )
        extension_interpreter.allocate_tensors()

    outputs = []
    acc_counter = 0
    st = time.time()
    latencies = []
    with tqdm(total=len(x_test)) as bar:
        for x in x_test:
            latencies.append(time.time())

            if type_ == "keras":

                output = conv_base_model(x, training=False)

                output_reshaped = np.reshape(output, newshape=(1, 512))

                final_output = extension_model(output_reshaped, training=False)
            elif type_ == "tflite":
                conv_base_interpreter.set_tensor(conv_base_interpreter.get_input_details()[0]["index"], x)
                conv_base_interpreter.invoke()
                output = conv_base_interpreter.get_tensor(conv_base_interpreter.get_output_details()[0]["index"])
                output_reshaped = np.reshape(output, newshape=(1, 512))
                extension_interpreter.set_tensor(extension_interpreter.get_input_details()[0]["index"], output_reshaped)
                extension_interpreter.invoke()
                final_output = extension_interpreter.get_tensor(extension_interpreter.get_output_details()[0]["index"])
            else:
                raise RuntimeError("invalid 'type_'")

            latencies[-1] = time.time() - latencies[-1]
            outputs += [final_output]
            bar.update()
        bar.close()
        exec_time = time.time() - st

    for output, y in zip(outputs, y_test):
        # print(output, y)
        if np.argmax(output) == y:
            acc_counter += 1

    return acc_counter / len(x_test), exec_time, np.mean(latencies)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    data = get_data()

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

    # run .tflite interpreter split
    result, accuracy, exec_time, avg_latency = main(num_stages, assignments, data, "interpreter")
    print("multi-interpreter .tflite exec.time: ", exec_time)
    print("multi-interpreter .tflite accuracy: ", accuracy)
    print("multi-interpreter .tflite avg_latency: ", avg_latency)
    # print("Result:", result[:10])

    # run model layer split
    result, accuracy, exec_time, avg_latency = main(num_stages, assignments, data, "layers")
    print("model layers split exec.time: ", exec_time)
    print("model layers split accuracy: ", accuracy)
    print("model layers split avg_latency: ", avg_latency)

    # run full models .keras
    acc, exec_time, avg_latency = run_two_split_model(data, 0, "keras")
    print("full keras models accuracy: ", acc)
    print("full keras models exec.time: ", exec_time)
    print("full keras models avg_latencies: ", avg_latency)

    # run full model_interpreters .tflite
    acc, exec_time, avg_latency = run_two_split_model(data, 0, "tflite")
    print("full tflite interpreters accuracy: ", acc)
    print("full tflite interpreters exec.time: ", exec_time)
    print("full tflite interpreters avg_latencies: ", avg_latency)

    # run full models .keras (tf in-build predicts and evaluates)
    print("use tensorflow inbuild predict function")
    input_ = np.array([np.reshape(x, (32, 32, 3)) for x in data[0]])
    inter_results = conv_base_model.predict(input_)
    inter_results = np.reshape(inter_results, (len(input_), 1 * 1 * 512))
    results = extension_model.evaluate(inter_results, data[1])
