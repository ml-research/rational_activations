from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Sequential
from rational.keras import Rational as KerasRational
from tensorflow import keras
import numpy as np


def prepare_data_keras(seed=4242, batch_size=256):
    # model / data parameters
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    train_data = ImageDataGenerator(featurewise_std_normalization=True, rotation_range=30, rescale=1.0 / 255)
    test_data = ImageDataGenerator(featurewise_std_normalization=True, rescale=1.0 / 255)

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train_data.fit(x_train, seed=seed)
    test_data.fit(x_test, seed=seed)
    train_generator = train_data.flow(x_train, y_train, batch_size=batch_size, seed=seed)
    test_generator = test_data.flow(x_test, y_test, batch_size=2 * batch_size, seed=seed)
    return train_generator, test_generator


def compile_keras_model(rational, version='A', approx_func='relu', input_shape=(28, 28, 1),
                        kernel_size=(3, 3), stride=(1, 1), padding='same', pooling=(2, 2)):
    model = Sequential([
        Input(shape=input_shape),
        Resizing(32, 32),
        Conv2D(filters=64, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        MaxPool2D(pool_size=pooling, strides=2),
        Conv2D(filters=128, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        MaxPool2D(pool_size=pooling, strides=2),
        Conv2D(filters=256, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        Conv2D(filters=256, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        MaxPool2D(pool_size=pooling, strides=2),
        Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        MaxPool2D(pool_size=pooling, strides=2),
        Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        Conv2D(filters=512, kernel_size=kernel_size, strides=stride, padding=padding),
        KerasRational(version=version, approx_func=approx_func) if rational else Activation('relu'),
        MaxPool2D(pool_size=pooling, strides=2),
        Flatten(),
        Dense(units=10, activation='softmax')
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=1e-2, momentum=0.5, clipnorm=5.0),
                  metrics=["accuracy"])
    return model
