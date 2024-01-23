import numpy as np
import tensorflow as tf
from tqdm import tqdm


def get_data():
    x_train = np.load("../mnist_dataset/non_pp_train.npz")["arr_0"]
    y_train = np.load("../mnist_dataset/non_pp_train.npz")["arr_1"]

    x_test = np.load("../mnist_dataset/non_pp_test.npz")["arr_0"]
    y_test = np.load("../mnist_dataset/non_pp_test.npz")["arr_1"]

    x_val = np.load("../mnist_dataset/non_pp_val.npz")["arr_0"]
    y_val = np.load("../mnist_dataset/non_pp_val.npz")["arr_1"]

    x_test = [x[None, :, :, :] for x in x_test]
    # print(x_train, y_train)
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


(x_train, y_train), (x_test, y_test), (x_val, y_val) = get_data()

interpreter = tf.lite.Interpreter("./alex_net_model.tflite")
interpreter.allocate_tensors()
acc_counter = 0
with tqdm(total=len(x_test)) as bar:
    for input_, label in zip(x_test, y_test):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_)
        interpreter.invoke()
        input_ = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        if label == np.argmax(input_):
            acc_counter += 1
        bar.update()
    bar.close()
print("Accuracy: ", acc_counter/len(x_test))


interpreters = []
for i in range(21):
    interpreter = tf.lite.Interpreter("./alex_net_model_" + str(i) + ".tflite")
    interpreter.allocate_tensors()
    interpreters.append(interpreter)

acc_counter = 0
with tqdm(total=len(x_test)) as bar:
    for input_, label in zip(x_test, y_test):
        for interpreter in interpreters:
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_)
            interpreter.invoke()
            input_ = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        if label == np.argmax(input_):
            acc_counter += 1
        bar.update()
    bar.close()
print("Accuracy: ", acc_counter/len(x_test))
