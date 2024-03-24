import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import config


def load_keras_model():
    # The naming convention is 'my_model.keras'. Windows views that as a file instead of a directory.
    # Both the standard and custom naming conventions will be used when saving model during training.
    # If keras versions change the linux version may also need
    if os.path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)
        x_shape = img_to_array(load_img(x[0])).shape
        model_location = os.path.join(config.windows_model_location, config.model_dir_name, config.model_windows_name)
        #model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    else:
        x, y = get_files(config.linux_testing_directory)
        model_location = os.path.join(config.linux_model_location, config.model_dir_name, config.model_windows_name)
        # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    return model, x, y


def get_files(recording_base_dir):
    """
    TODO: use glob instead for easier code comprehension
    Cannot load the image files into memory. Too large.
    :return: list of files, json keypress file
    """
    frame_dir_list = []
    df_combined = pd.DataFrame()

    recordings = os.listdir(recording_base_dir)
    for session_dir in recordings:
        recording_dir = os.path.join(recording_base_dir, session_dir)

        video_frame_dir = os.path.join(recording_dir, "video_images")
        frame_list = os.listdir(video_frame_dir)

        frame_dir_list.extend([os.path.join(video_frame_dir, frame) for frame in frame_list])

        recording_file = 'inputs.csv'
        df = pd.read_csv(os.path.join(recording_dir, recording_file))
        df = df.drop(columns=df.columns[0], axis=1)
        df_combined = pd.concat([df_combined, df])

    return frame_dir_list, df_combined
