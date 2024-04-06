"""
Non-trainable models used for visualization
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D)

import tensorflow.keras.layers as layers

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config


def build_test_cnn(input_shape):
    """
    Model used for visualizing CNNs
    """
    new_model = Sequential([
        Conv2D(filters=8, kernel_size=(8, 8), padding="same", activation='relu', input_shape=input_shape),
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


def build_test_merged_cnn(input_shape):
    """
    Model used for visualizing CNNs
    """
    inputs = layers.Input(shape=input_shape)
    model_a = Conv2D(filters=2, kernel_size=(16, 16), padding="same", activation='relu', input_shape=input_shape)(inputs)
    model_b = Conv2D(filters=2, kernel_size=(4, 4), padding="same", activation='relu', input_shape=input_shape)(inputs)
    merged = layers.concatenate([model_a, model_b], axis=-1)

    new_model = Model(inputs=inputs, outputs=merged)

    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_3_filter_cnn_with_speed(input_shape):
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

    speed = Dense(32, activation='relu')(f)

    image_with_speed = layers.concatenate([f, speed], axis=-1)

    d1 = Dense(32, activation='relu')(image_with_speed)
    output = Dense(4, activation='sigmoid')(d1)

    new_model = Model(inputs=inputs, outputs=output)

    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def build_merged_cnn_3_filters(input_shape):
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



def build_merge_two_models():

