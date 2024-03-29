import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config


def build_conv_lstm(input_shape):
    print((config.gpu_batch, *input_shape))
    new_model = Sequential([
        ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=False,
            activation="relu",
            input_shape=(config.gpu_batch, *input_shape)
        ),
        BatchNormalization(),
        # ConvLSTM2D(
        #     filters=64,
        #     kernel_size=(3, 3),
        #     padding="same",
        #     return_sequences=False,
        #     activation="relu",
        # ),
        # BatchNormalization(),
        # ConvLSTM2D(
        #     filters=64,
        #     kernel_size=(1, 1),
        #     padding="same",
        #     return_sequences=False,
        #     activation="relu",
        # ),
        Dense(4, activation='sigmoid'),
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    return new_model


def build_cnn_model(input_shape):
    """
    Build new model based on input image.
    :return: TensorFlow Keras model
    """
    new_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_cnn_lstm_model(input_shape):
    """
        Build new model based on input image.
        :return: TensorFlow Keras model
    """
    # new_model = Sequential([
    #     TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='tanh'),
    #                     input_shape=(gpu_batch, 720, 1280, 1)),
    #     # ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', input_shape=(720, 1280, 1)),
    #     # MaxPooling2D(pool_size=(2, 2)),
    #     # ConvLSTM2D(filters=32, kernel_size=(4, 4), activation='tanh'),
    #     # MaxPooling2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dense(32, activation='relu'),
    #     # 4 labels.
    #     Dense(4, activation='sigmoid')
    # ])

    new_model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(32),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_test_cnn(input_shape):
    """
    Model used for visualizing CNNs
    """
    new_model = Sequential([
        Conv2D(filters=12, kernel_size=(16, 9), activation='relu', input_shape=input_shape),
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_test_cnn_with_pooling(input_shape):
    """
    Model used for Visualizing max pooling
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_test_cnn_model(input_shape):
    """
    Model used for Visualizing max pooling
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model
