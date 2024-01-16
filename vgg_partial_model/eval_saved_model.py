import numpy as np
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


base_model = tf.keras.models.load_model("vgg_conv_base_model/conv_base_model.keras")
ext_model = tf.keras.models.load_model("vgg_extension_model/extension_model.keras")
(x_train, y_train), (x_test, y_test), (x_val, y_val) = get_data()


inter_results = base_model.predict(x_test)
inter_results = np.reshape(inter_results, (len(x_test), 1 * 1 * 512))

results = ext_model.evaluate(inter_results, y_test)

#final_results = ext_model.predict(inter_results)
#print(final_results, y_test)

