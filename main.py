import argparse
import multiprocessing
import time

import numpy as np
import psutil
import tensorflow as tf
from tqdm import tqdm

NUM_TEST_DATA = 1000

def get_data(model):
    if model == "VGG":
        x_train = np.load("./mnist_dataset/train.npz")["arr_0"]
        y_train = np.load("./mnist_dataset/train.npz")["arr_1"]

        x_test = np.load("./mnist_dataset/test.npz")["arr_0"]
        y_test = np.load("./mnist_dataset/test.npz")["arr_1"]

        x_val = np.load("./mnist_dataset/val.npz")["arr_0"]
        y_val = np.load("./mnist_dataset/val.npz")["arr_1"]

    elif model == "ALEX_NET":
        x_train = np.load("./mnist_dataset/non_pp_train.npz")["arr_0"]
        y_train = np.load("./mnist_dataset/non_pp_train.npz")["arr_1"]

        x_test = np.load("./mnist_dataset/non_pp_test.npz")["arr_0"]
        y_test = np.load("./mnist_dataset/non_pp_test.npz")["arr_1"]

        x_val = np.load("./mnist_dataset/non_pp_val.npz")["arr_0"]
        y_val = np.load("./mnist_dataset/non_pp_val.npz")["arr_1"]
    else:
        raise RuntimeError("wrong model")

    x_test = [x[None, :, :, :] for x in x_test]

    return x_test[:NUM_TEST_DATA], y_test[:NUM_TEST_DATA]


def get_interpreters(partial_model_dir_path, model_name, size):
    interpreters = []
    for i in range(size):
        interpreter = tf.lite.Interpreter(partial_model_dir_path + "/" + model_name + "_" + str(i) + ".tflite")
        interpreter.allocate_tensors()
        interpreters.append(interpreter)
    return interpreters


def init_interpreters():
    # if model == "VGG":
    interpreters = get_interpreters(partial_model_dir_path="vgg_partial_model/vgg_conv_base_model",
                                    model_name="conv_base_model",
                                    size=19
                                    )
    interpreters += get_interpreters(partial_model_dir_path="vgg_partial_model/vgg_extension_model",
                                     model_name="extension_model",
                                     size=4
                                     )
    return interpreters


interpreters = init_interpreters()


def invoke_vgg_model_interpreters(stage, input_):
    interpreter = interpreters[stage]
    st = time.time()
    if stage == 19:
        input_ = np.reshape(input_, newshape=(1, 512))
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    et = time.time() - st
    return output, et


def invoke_alex_net_model_interpreters(stage, input_, interpreter):
    st = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    et = time.time() - st
    return output, et


interpreter_invoke_funcs = {
    "VGG": invoke_vgg_model_interpreters,
    "ALEX_NET": invoke_alex_net_model_interpreters,
}


def run_stage(stage_idx, input_, model):
    return interpreter_invoke_funcs[model](stage_idx, input_)


# Function to simulate a pipeline stage
def assignment_worker(in_queue, out_queue, assignment, core_id, latencies, model):
    # Set CPU affinity for this process
    st = time.time()
    psutil.Process().cpu_affinity([core_id])
    # print("aff. time: ", time.time() - st)

    counter = 0
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
            #print(stage, data, model)
            #print(np.shape(data))
            data, et = run_stage(stage, data, model)
            latencies[stage * NUM_TEST_DATA + counter] = et
        counter += 1
        out_queue.put(data)


def main(data, assignments, model):
    if model == "VGG":
        num_stages = 23
    elif model == "ALEX_NET":
        num_stages = 21
    else:
        raise RuntimeError("INVALID MODEL NAME: ", model)

    # Create queues for communication between stages
    queues = [multiprocessing.Queue() for _ in range(len(assignments) + 1)]
    latencies = multiprocessing.Array('d', NUM_TEST_DATA * num_stages)

    print(assignments)

    # Create processes for each stage
    processes = [
        multiprocessing.Process(target=assignment_worker,
                                args=(
                                    queues[idx], queues[idx + 1], assignment, core_id, latencies, model))
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

    latencies_ = [np.sum([latencies[stage * NUM_TEST_DATA + counter] for stage in range(num_stages)])
                  for counter in range(NUM_TEST_DATA)]
    avg_latency = np.mean(latencies_)

    return results, acc_counter / len(y_test), et, avg_latency


'''
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

                output = conv_base_model(x)

                output_reshaped = np.reshape(output, newshape=(1, 512))

                final_output = extension_model(output_reshaped)
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
'''


def func1(data, assignments, model):
    # run .tflite interpreter split
    result, accuracy, exec_time, avg_latency = main(data, assignments, model)
    print("multi-interpreter .tflite exec.time: ", exec_time)
    print("multi-interpreter .tflite accuracy: ", accuracy)
    print("multi-interpreter .tflite avg_latency: ", avg_latency)
    # print("Result:", result[:10])


'''
def func2():
    # run model layer split
    result, accuracy, exec_time, avg_latency = main(num_stages, assignments, data, "layers")
    print("model layers split exec.time: ", exec_time)
    print("model layers split accuracy: ", accuracy)
    print("model layers split avg_latency: ", avg_latency)


def func3():
    # run full models .keras
    acc, exec_time, avg_latency = run_two_split_model(data, 0, "keras")
    print("full keras models accuracy: ", acc)
    print("full keras models exec.time: ", exec_time)
    print("full keras models avg_latencies: ", avg_latency)


def func4():
    # run full model_interpreters .tflite
    acc, exec_time, avg_latency = run_two_split_model(data, 0, "tflite")
    print("full tflite interpreters accuracy: ", acc)
    print("full tflite interpreters exec.time: ", exec_time)
    print("full tflite interpreters avg_latencies: ", avg_latency)


def func5():
    # run full models .keras (tf in-build predicts and evaluates)
    print("use tensorflow inbuild predict function")
    input_ = np.array([np.reshape(x, (32, 32, 3)) for x in data[0]])
    inter_results = conv_base_model.predict(input_)
    inter_results = np.reshape(inter_results, (len(input_), 1 * 1 * 512))
    results = extension_model.evaluate(inter_results, data[1])
'''


def wrapper(data, assignment, model):
    # funcs = [func1, func2, func3, func4, func5]
    # funcs[i]()
    func1(data, assignment, model)


def get_active_core_range(ind, core_id, num_stages):
    starting = -1
    ending = -1
    for stage_idx, core_idx in enumerate(ind):
        if core_idx == core_id and starting < 0:
            starting = stage_idx
        if core_idx != core_id and starting >= 0 and ending < 0:
            ending = stage_idx - 1
    if starting == -1:
        starting = num_stages - 1
    if ending == -1:
        ending = num_stages - 1
    return starting, ending


def ind_to_assignment(ind):
    assignment = {}
    for core_id in ind:
        starting, ending = get_active_core_range(ind, core_id, len(ind))
        assignment[core_id] = list(range(starting, ending + 1))

    return assignment


if __name__ == "__main__":
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # print(tf.__version__)

    parser = argparse.ArgumentParser(description="Pipeline Parallelism")
    parser.add_argument("-ind", "--individual", nargs='+', type=int, help="Assignment of cores to stages", required=True)
    parser.add_argument("-i", "--index", type=int, help="Index for certain benchmark", required=True)
    parser.add_argument("-m", "--model", type=str, help="Model (VGG OR ALEX_NET)", required=True)
    args = parser.parse_args()

    individual = args.individual
    i = args.index
    model = args.model

    assert len(individual) == 23

    data = get_data(model)

    assignment = ind_to_assignment(individual)

    wrapper(data, assignment, model)
