import tensorflow as tf
from os import path
from train import get_files, build_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config
import pandas as pd
import gc
import numpy as np
import dxcam
import time
import keyboard


def validate_model(model, x_test, y_test):
    memory_batch = 256
    validation_rounds = 0
    accumulative_accuracy = 0

    while len(x_test) > 0:
        current_x_train = []
        current_y_train = []
        for _ in range(memory_batch):
            if len(x_test) == 0:
                break
            current_x_train.append(img_to_array(load_img(x_test.pop(0))) / 255)
            current_y_train.append(y_test.pop(0))

        current_x_train = np.array(current_x_train)
        current_y_train = np.array(current_y_train)

        test_loss, test_accuracy = model.evaluate(current_x_train, current_y_train, verbose=2)
        print(f'Test accuracy: {test_accuracy * 100:.2f}%')

        accumulative_accuracy += test_accuracy
        validation_rounds += 1

        del current_x_train
        del current_y_train
        gc.collect()
    print(f'total accuracy: {accumulative_accuracy / validation_rounds}')


def visualize_prediction(model, x_test, y_test):
    for i in range(len(x_test)):
        single_x = img_to_array(load_img(x_test[i]))
        # normally our datashape is (num_pictures, width, height, depth)
        # only one image will be (width, height, depth)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        print(single_x.shape)
        print(model.predict(single_x))
        print('actual', y_test.iloc[i])


def inference_on_game(model):
    camera = dxcam.create(device_idx=0, output_idx=1)
    camera.start(region=(0, 0, 1280, 720), target_fps=30)
    model_run = False
    current_keys = []

    print("ready for model to drive")
    while True:
        if keyboard.is_pressed('f9'):
            print('model running')
            model_run = True
        if keyboard.is_pressed('esc'):
            print('program killed by user')
            exit()
        if model_run:
            # inference on CPU takes too long.
            # keyboard.release('w')
            # keyboard.release('a')
            # keyboard.release('s')
            # keyboard.release('d')

            frame = camera.get_latest_frame()
            frame = frame.reshape(-1, frame.shape[0], frame.shape[1], frame.shape[2])
            predicted_keys = model.predict(frame)
            predicted_keys = predicted_keys[0].tolist()
            if predicted_keys[0] > 0.5:
                keyboard.press('w')
            else:
                keyboard.release('w')
            if predicted_keys[1] > 0.5:
                keyboard.press('a')
            else:
                keyboard.release('a')
            if predicted_keys[2] > 0.5:
                keyboard.press('s')
            else:
                keyboard.release('s')
            if predicted_keys[3] > 0.5:
                keyboard.press('d')
            else:
                keyboard.release('d')
            print(predicted_keys)


if __name__ == "__main__":
    # Turn on to run model.validate on testing data, otherwise run inference on live game.
    validate = False

    # The naming convention is 'my_model.keras'. Windows views that as a file instead of a directory.
    # Both the standard and custom naming conventions will be used when saving model during training.
    if path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)
        x_shape = img_to_array(load_img(x[0])).shape
        model_location = path.join(config.windows_model_location, "current_model", "keras_model_dir")
        #model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    else:
        x, y = get_files(config.linux_testing_directory)
        model_location = path.join(config.linux_model_location, "current_model", config.model_name)
        # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)

    if validate:
        validate_model(model, list(x), y.values.tolist())
    else:
        inference_on_game(model)

