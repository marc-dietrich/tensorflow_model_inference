import numpy as np
import psutil
import tensorflow.lite as tflite
from tqdm import tqdm

BATCH_SIZE = 32
NUM_BATCHES = 300

# Load your TFLite model
interpreter = tflite.Interpreter(model_path="alex_net_model.tflite")
interpreter.resize_tensor_input(0, [BATCH_SIZE, 32, 32, 3])
interpreter.allocate_tensors()


def data():
    # x_train = np.load("../mnist_dataset/non_pp_train.npz")["arr_0"]
    # y_train = np.load("../mnist_dataset/non_pp_train.npz")["arr_1"]

    x_test = np.load("../mnist_dataset/non_pp_test.npz")["arr_0"]
    y_test = np.load("../mnist_dataset/non_pp_test.npz")["arr_1"]

    # x_val = np.load("../mnist_dataset/non_pp_val.npz")["arr_0"]
    # y_val = np.load("../mnist_dataset/non_pp_val.npz")["arr_1"]
    return x_test[:BATCH_SIZE * NUM_BATCHES], y_test[:BATCH_SIZE * NUM_BATCHES]


# Assuming your input tensor index is 0 (modify if necessary)
input_tensor_index = 0

features, y_features = data()

psutil.Process().cpu_affinity([0])

# Iterate over your data in batches
with tqdm(total=len(features)) as bar:
    num_splits = features.shape[0] // BATCH_SIZE
    # Use np.split to perform the split
    split_arrays = np.split(features, num_splits, axis=0)

    for input_data in split_arrays:
        interpreter.set_tensor(input_tensor_index, input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor data
        output_tensor_index = 0  # Assuming output tensor index is 0, modify if necessary
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        # Process output data as needed

        # print(output_data.shape)
        '''
        acc_counter = 0
        for x, y in zip(output_data, y_features[batch_start:batch_start + 32]):
            if y == np.argmax(x):
                acc_counter += 1
                '''
        bar.update(32)
    bar.close()

#accuracy = acc_counter / len(features)
#print(accuracy)

'''
    for batch_start in range(0, len(features), 32):
        batch_features = features[batch_start:batch_start + 32]
        # Assuming your input shape is [32, height, width, channels]
        # You may need to modify this according to your model
        input_shape = interpreter.get_input_details()[input_tensor_index]['shape']
        input_data = np.zeros(input_shape, dtype=np.float32)
        input_data[:len(batch_features)] = batch_features

        #print(input_data.shape)

        # Set input tensor data
'''
