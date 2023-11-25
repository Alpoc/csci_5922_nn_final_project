import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# import seaborn as sns
import pandas as pd
import os
import cv2
import json
import imageio.v3 as imageio


def get_files():
    """
    Cannot load the image files into memory. Too large.
    :return: list of files, json keypress file
    """
    video_frames = []

    recording_base_dir = "recordings"
    recordings = os.listdir(recording_base_dir)
    for session_dir in recordings:
        recording_dir = os.path.join(recording_base_dir, session_dir)

        video_frame_dir = os.path.join(recording_dir, "video_images")
        frame_list = os.listdir(video_frame_dir)

        frame_dir_list = [os.path.join(video_frame_dir, frame) for frame in frame_list]


        # for frame in os.listdir(video_frame_dir):
        #     np_frame = imageio.imread(os.path.join(video_frame_dir, frame))
        #     video_frames.append(np_frame)

        recording_file = 'inputs.json'
        with open(os.path.join(recording_dir, recording_file), 'r') as key_file:
            key_presses = json.load(key_file)
            print(f'num key presses: {len(key_presses)}')

    return np.array(frame_dir_list), key_presses


def train(x_train, y_train):
    pass
    # print(f'x train shape: {x_train.shape}')
    # print(f'y train shape: {y_train.shape}')
    # print(f'x test shape: {x_test.shape}')
    # print(f'y test shape: {y_test.shape}')
    #
    # model = Sequential([
    #     # input shape was only giving 28, 28. It was missing the 1 channel for greyscale
    #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(filters=64, kernel_size=(4, 2), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dense(64, activation='relu'),
    #     # 10 labels.
    #     Dense(10, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model_hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    #
    # plt.plot(model_hist.history['accuracy'], label='accuracy')
    # plt.plot(model_hist.history['loss'], label='loss')
    # plt.legend()
    # plt.show()
    # input('press key to continue')
    #
    # test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    # print(f'Test accuracy: {test_accuracy * 100:.2f}%')


def load_model_onto_gpu():
    with tf.device("/GPU:0"):
        x = imageio.imread(os.path.join(x_train[0]))
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                   input_shape=x.shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(4, 2), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            # 10 labels.
            Dense(10, activation='softmax')
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_in_batches(x_train, y_train, model):
    while len(x_train) > 0:
        current_x_train = []
        current_y_train = []
        for _ in range(1000):
            current_x_train.append(imageio.imread(x_train.pop(0)))
            current_y_train.append(imageio.imread(y_train.pop(0)))

        model_hist = model.fit(current_x_train, current_y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

if __name__ == "__main__":
    x, y = get_files()

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print(f"training set x: {len(x_train)}")
    print(f"training set y: {len(y_train)}")
    print(f"test set x: {len(x_test)}")
    print(f"test set y: {len(y_test)}")

    # Data Normalized to 0 to 1. Pixel values are 0-255
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    y_test = [imageio.imread(y) for y in y_test]

    model = load_model_onto_gpu()
    train_in_batches(x_train, y_train)