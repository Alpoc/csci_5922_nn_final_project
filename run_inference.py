import tensorflow as tf
from os import path
from train import get_files, build_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config
import pandas as pd


def validate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')


def visualize_prediction(model, x_test, y_test):
    for i in range(len(x_test)):
        x = img_to_array(load_img(x_test[i]))
        x = x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])
        print(x.shape)
        print(model.predict(x))
        print('actual', y_test.iloc[i])


if __name__ == "__main__":
    # gpu_check()
    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False

    if path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)

    else:
        print("loading model on linux")
        x, y = get_files(config.linux_testing_directory)
        model_location = path.join(config.linux_model_location, config.model_name)
        model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)

    # print('loading test images into memory')
    # x = [img_to_array(load_img(x)) / 255 for x in x]

    visualize_prediction(model, x, y)

    # model = build_model(x[0].shape)
    # model.set_weights(config.windows_model_location)
    # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)


    # validate_model(model, x, y)
