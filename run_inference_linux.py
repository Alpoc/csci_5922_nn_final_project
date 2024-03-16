import tensorflow as tf
from os import path
from train import get_files, build_cnn_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import config
import pandas as pd
import gc
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import pygame
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Lots off incompatable libraries on windows vs linux
ON_WINDOWS = False

if ON_WINDOWS:
    import keyboard
    import dxcam
else:
    from vidgear.gears import ScreenGear
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

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
            if grayscale:
                current_x_train.append(img_to_array(load_img(x_test.pop(0), color_mode='grayscale')) / 255)
            else:
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


def conf_model(model, x_test, y_test):
    predictions = []
    for i in range(len(x_test)):
        single_x = img_to_array(load_img(x_test[i], color_mode="grayscale")) / 255
        # normally our datashape is (num_pictures, width, height, depth)
        # only one image will be (width, height, depth)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        predictions.append(model.predict(single_x, verbose=False)[0])

    # Confusion Matrix
    y_pred = np.argmax(predictions, axis=1)

    y_pred_labels = []
    for index, row in y_test.iterrows():
        y_pred_labels.append(row.argmax())

    conf_matrix = confusion_matrix(y_pred_labels, y_pred)

    # Plot confusion matrix
    class_names = list(set(y_test))
    df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def visualize_prediction(model, x_test, y_test):
    for i in range(len(x_test)):
        single_x = img_to_array(load_img(x_test[i], color_mode="grayscale")) / 255
        # normally our datashape is (num_pictures, width, height, depth)
        # only one image will be (width, height, depth)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        print(single_x.shape)
        print(model.predict(single_x))
        print('actual', y_test.iloc[i])


def inference_on_game(model):
    if ON_WINDOWS:
        camera = dxcam.create(device_idx=0, output_idx=1)
        camera.start(region=(0, 0, 1280, 720), target_fps=30)
    else:
        options = {'top': 0, 'left': 0, 'width': 1280, 'height': 720}
        stream = ScreenGear(framerate=30, backend="pil", window="1", **options).start()
    model_run = False

    print("ready for model to drive")
    while True:
        # if keyboard.is_pressed('f9'):
        #     print('model running')
        #     model_run = True
        # if keyboard.is_pressed('esc'):
        #     print('program killed by user')
        #     exit()
        # if True:
        #     # show screen for test
        #     if ON_WINDOWS:
        #         frame = camera.get_latest_frame()
        #     else:
        #         frame = stream.read()
        #     gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #     cv2.imshow("image", gray_image)
        #     cv2.waitKey(0)

        if True:
            if ON_WINDOWS:
                frame = camera.get_latest_frame()
            else:
                frame = stream.read()
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255
                frame = frame.reshape(-1, frame.shape[0], frame.shape[1])
            else:
                frame = frame.reshape(-1, frame.shape[0], frame.shape[1], frame.shape[2])
            start_time = time.time()
            if average_results:
                prediction_1 = []
                for _ in range(3):

                    predicted_keys = model.predict(frame, verbose=0) / 2
                    if len(prediction_1) == 0:
                        prediction_1 = predicted_keys.copy()
                predicted_keys = list(map(sum,zip(prediction_1, predicted_keys)))
            else:
                predicted_keys = model.predict(frame, verbose=0)
            print(f'inferent time {time.time() - start_time}')
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
            # print(predicted_keys)
        clock.tick_busy_loop(60)


if __name__ == "__main__":
    # Turn on to run model.validate on testing data, otherwise run inference on live game.
    validate = False
    grayscale = True
    # run two inference steps and average the results together.
    average_results = True

    pygame.init()
    clock = pygame.time.Clock()

    # The naming convention is 'my_model.keras'. Windows views that as a file instead of a directory.
    # Both the standard and custom naming conventions will be used when saving model during training.
    if path.exists(config.windows_testing_directory):
        x, y = get_files(config.windows_testing_directory)
        x_shape = img_to_array(load_img(x[0])).shape
        model_location = path.join(config.windows_model_location, "current_model_cnn_100_epochs", "keras_model_dir")
        #model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)
    else:
        x, y = get_files(config.linux_testing_directory)
        model_location = path.join(config.linux_model_location, "current_model_cnn_100_epochs", "keras_model_dir")
        # model_location = path.join(config.linux_model_location, "current_model_cnn_100_epochs", config.model_name)
        # model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    model = tf.keras.models.load_model(model_location, custom_objects=None, compile=True)

    # model.summary()
    # exit()

    # conf_model(model, x, y)
    #visualize_prediction(model, x, y)

    if validate:
        validate_model(model, list(x), y.values.tolist())
    else:
        inference_on_game(model)

