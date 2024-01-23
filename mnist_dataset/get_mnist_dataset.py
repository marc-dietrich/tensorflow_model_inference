import numpy as np
import tensorflow as tf
from keras import datasets
from keras.applications.vgg16 import preprocess_input

''' preprocess data '''
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
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

# datas = [x_train, x_test, x_val, y_train, y_test, y_val]
# for d in datas:
#    print(np.shape(d))

np.savez("non_pp_train", x_train, y_train)
np.savez("non_pp_test", x_test, y_test)
np.savez("non_pp_val", x_val, y_val)

quit()

x_train, x_test, x_val = preprocess_input(x_train), preprocess_input(x_test), preprocess_input(x_val)

np.savez("train", x_train, y_train)
np.savez("test", x_test, y_test)
np.savez("val", x_val, y_val)


