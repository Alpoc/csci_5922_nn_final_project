import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import pandas as pd
import os
import cv2
import json


def get_files():
    x = []
    y = []
    recordings = os.listdir("video_images")
    for recording_dir in recordings:
        video_pngs = os.listdir(os.path.join("video_images", recording_dir))
        print(f'num frames: {len(video_pngs)}')

    recording_file = 'controller_recording_2023_11_24-12_25_17.json'
    with open(os.path.join('controller_recordings', recording_file), 'r') as key_file:
        key_presses = json.load(key_file)
        print(f'num key presses: {len(key_presses)}')


# Data Normalized to 0 to 1. Pixel values are 0-255
# x_train = x_train / 255.0
# x_test = x_test / 255.0

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


if __name__ == "__main__":
    get_files()