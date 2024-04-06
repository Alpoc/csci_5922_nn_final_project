import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D)

import tensorflow.keras.layers as layers

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





def build_merged_cnn(input_shape):
    """
    Model used for visualizing CNNs
    """

    inputs = layers.Input(shape=input_shape)

    model_a = Conv2D(filters=2, kernel_size=(8, 8), padding="same", activation='relu', input_shape=input_shape)(inputs)
    model_a = layers.AveragePooling2D(pool_size=(2, 2))(model_a)

    model_b = Conv2D(filters=2, kernel_size=(4, 4), padding="same", activation='relu', input_shape=input_shape)(inputs)
    model_b = layers.AveragePooling2D(pool_size=(2, 2))(model_b)
    merged = layers.concatenate([model_a, model_b], axis=-1)

    model_c = Conv2D(filters=2, kernel_size=(3, 3), padding="same", activation='relu', input_shape=input_shape)(inputs)
    model_c = layers.AveragePooling2D(pool_size=(2, 2))(model_c)
    merged = layers.concatenate([merged, model_c], axis=-1)

    f = Flatten()(merged)
    d1 = Dense(32, activation='relu')(f)
    output = Dense(4, activation='sigmoid')(d1)

    new_model = Model(inputs=inputs, outputs=output)

    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model



def build_cnn_model_32x5_nn(input_shape):
    """
    CNN model with 32x5 NN following the conv
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_cnn_model_64x10_nn(input_shape):
    """
    CNN model with 64x5 NN following the conv
    Too large for my 3090
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_cnn_model_48x10_nn(input_shape):
    """
    CNN model with 64x5 NN following the conv
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        Dense(48, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model
