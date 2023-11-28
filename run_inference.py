import tensorflow as tf
from os import path
from train import get_files, build_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config
import pandas as pd
import numpy as np
import gc
import PIL

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


if __name__ == "__main__":
    # gpu_check()
    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False

    if path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)
        x_shape = img_to_array(load_img(x[0])).shape
        model_location = path.join(config.windows_model_location, "current_model", "keras_model_dir")

        model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)

        # weights_location = path.join(config.windows_model_location, "current_model",
        #                              "checkpoint")
        # model = build_model(x_shape)
        # model.set_weights(weights_location)
        # model.load_weights(weights_location)


    else:
        x, y = get_files(config.linux_testing_directory)
        model_location = path.join(config.linux_model_location, "current_model", config.model_name)
        model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)

    validate_model(model, list(x), y.values.tolist())
