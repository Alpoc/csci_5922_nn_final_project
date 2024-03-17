from os import path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from train import get_files
import config


def load_keras_model():
    # The naming convention is 'my_model.keras'. Windows views that as a file instead of a directory.
    # Both the standard and custom naming conventions will be used when saving model during training.
    # If keras versions change the linux version may also need
    if path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)
        x_shape = img_to_array(load_img(x[0])).shape
        model_location = path.join(config.windows_model_location, config.model_dir_name, config.model_windows_name)
        #model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    else:
        x, y = get_files(config.linux_testing_directory)
        model_location = path.join(config.linux_model_location, config.model_dir_name, config.model_windows_name)
        # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    return model, x, y
