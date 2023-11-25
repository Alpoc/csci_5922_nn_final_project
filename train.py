import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# import seaborn as sns
import pandas as pd
import os
import cv2
import json
import imageio.v3 as imageio
from sklearn.preprocessing import MultiLabelBinarizer


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

        recording_file = 'inputs.csv'
        # with open(os.path.join(recording_dir, recording_file), 'r') as key_file:
        #     key_presses = json.load(key_file)
        #     print(f'num key presses: {len(key_presses)}')
        df = pd.read_csv(os.path.join(recording_dir, recording_file))
        df = df.drop(columns=df.columns[0], axis=1)
        # mlb = MultiLabelBinarizer()
        # df["encoded_labels"] = mlb.fit(df)

        # first column is index so drop it.
        # key_presses.drop(columns=key_presses.columns[0], axis=1, inplace=True)
        # # remove labels row
        # key_presses.iloc[1:, :]

    return frame_dir_list, df


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

    # x = imageio.imread(x_train[0]))
    # sample_x = cv2.imread(x_train[0])
    sample_x = img_to_array(load_img(x_train[0]))
    gpu_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=sample_x.shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        # 4 labels.
        Dense(4, activation='softmax')
        # tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    ])
    gpu_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return gpu_model


def train_in_batches(x_train, y_train, model, x_test, y_test):
    """
    We cannot load all of the frames into memory so do it incrementally.
    :param x_train:
    :param y_train:
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    while len(x_train) > 0:
        current_x_train = []
        current_y_train = []
        for _ in range(1000):
            if len(x_train) == 0:
                break
            # current_x_train.append(imageio.imread(x_train.pop(0)) / 255)
            # current_x_train.append(cv2.imread(x_train.pop(0)))
            current_x_train.append(img_to_array(load_img(x_train.pop(0))))
            current_y_train.append(y_train.pop(0))
        current_x_train = np.array(current_x_train)

        with tf.device("/GPU:0"):
            model_hist = model.fit(current_x_train, np.array(current_y_train), epochs=5, batch_size=1000, validation_data=(x_test, y_test))


def test_gpu():
    print(tf.config.experimental.list_physical_devices('GPU'))


if __name__ == "__main__":
    test_gpu()
    exit()
    x, y = get_files()

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print(f"training set x: {len(x_train)}")
    print(f"training set y: {len(y_train)}")
    print(f"test set x: {len(x_test)}")
    print(f"test set y: {len(y_test)}")

    # Data Normalized to 0 to 1. Pixel values are 0-255
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # x_test = [imageio.imread(x) / 255 for x in x_test]
    # x_test = [cv2.imread(x) / 255 for x in x_test]
    x_test = np.array([img_to_array(load_img(x)) for x in x_test])

    model = load_model_onto_gpu()
    train_in_batches(list(x_train), y_train.values.tolist(), model, x_test, y_test.to_numpy())