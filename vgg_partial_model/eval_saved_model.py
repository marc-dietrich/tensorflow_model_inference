import argparse
import os
import time

import numpy as np
import psutil
import tensorflow as tf


def get_data():
    x_train = np.load("../mnist_dataset/train.npz")["arr_0"]
    y_train = np.load("../mnist_dataset/train.npz")["arr_1"]

    x_test = np.load("../mnist_dataset/test.npz")["arr_0"]
    y_test = np.load("../mnist_dataset/test.npz")["arr_1"]

    x_val = np.load("../mnist_dataset/val.npz")["arr_0"]
    y_val = np.load("../mnist_dataset/val.npz")["arr_1"]

    # print(x_train, y_train)
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--num_cores", type=int, default=1, help="Number of CPU cores")
    parser.add_argument("--core_affinities", type=int, nargs="+", default=[6], help="Core affinities")

    args = parser.parse_args()

    # Specify the number of CPU cores and the core affinities
    num_cores = args.num_cores
    core_affinities = args.core_affinities  # Adjust based on your system

    # Set the number of intra and inter-operation threads
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)

    # Set CPU affinity for each thread
    #os.sched_setaffinity()   .sched_setaffinity(0, core_affinities)
    psutil.Process().cpu_affinity(core_affinities)


    base_model = tf.keras.models.load_model("vgg_conv_base_model/conv_base_model.keras")
    ext_model = tf.keras.models.load_model("vgg_extension_model/extension_model.keras")
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = get_data()

    num_samples = 10_000

    st = time.time()
    inter_results = base_model.predict(x_test[:num_samples])
    inter_results = np.reshape(inter_results, (num_samples, 1 * 1 * 512))

    results = ext_model.evaluate(inter_results, y_test[:num_samples])
    print("exec. time vgg_model precict function: ", time.time() - st)

    #final_results = ext_model.predict(inter_results)
    #print(final_results, y_test)

