import tensorflow as tf
from os import path
from train import get_files, build_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config


def validate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')


if __name__ == "__main__":
    # gpu_check()
    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False

    x, y = get_files(config.windows_testing_directory)
    print('loading test images into memory')
    x = [img_to_array(load_img(x)) / 255 for x in x]

    # model = build_model(x[0].shape)
    # model.set_weights(config.windows_model_location)
    # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)


    # validate_model(model, x, y)
