"""
This module contains various visualizations to try and get a glimpse into what the model is learning.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import gc

import build_models
from utils import load_keras_model, get_files

import config


def visualize_kernels(model):
    """
    Displays the kernels in the convolutional layers
    """
    # number of filters to visualize
    num_filters = 32
    print(f"model layers: {model.layers}")
    for layer in model.layers:
        if "conv" not in layer.name:
            continue

        fig = plt.figure(figsize=(16, 9))
        filters, bias = layer.get_weights()
        filters_min, filters_max = filters.min(), filters.max()
        filters = (filters - filters_min) / (filters_max - filters_min)
        for i in range(num_filters):
            f = filters[:, :, :, i]
            plt.subplot(8, 8, i + 1)
            plt.imshow(f[:, :, 0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()


def visualize_feature_map(model, x_test):
    """
    Loops through the provided model recreating the model with each conv layer displaying what the layer is seeing
    """
    single_visual = True

    plt.imshow(load_img(x_test[image_index]))

    for layer in model.layers:
        if "conv" not in layer.name:
            continue
        print(layer.name)
        # Recreates using a single layer model with the convolutional layer as the output layer.
        feature_model = Model(inputs=model.inputs, outputs=layer.output)

        single_x = img_to_array(load_img(x_test[image_index], color_mode="grayscale"))
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        features = feature_model.predict(single_x, verbose=False)

        plt.figure(figsize=(16, 9))
        if not single_visual:
            fig, ax = plt.subplots(figsize=(160, 90))
            fig.tight_layout()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        for i in range(1, features.shape[3] + 1):
            if not single_visual:
                plt.subplot(4, 3, i)
                plt.xticks([])
                plt.yticks([])
            plt.imshow(features[0, :, :, i-1], cmap="gray")
            if single_visual:
                plt.show()
                plt.figure(figsize=(16, 9))
        if not single_visual:
            plt.show()


def visualize_model(feature_model, x_test):
    """
    Similar to visualizing the feature map except we will use custom models to maintain features with pooling
    """
    single_visual = True

    single_x = img_to_array(load_img(x_test[image_index], color_mode="grayscale"))
    single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
    features = feature_model.predict(single_x, verbose=False)

    plt.figure(figsize=(16, 9))
    if not single_visual:
        fig, ax = plt.subplots(figsize=(160, 90))
        fig.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    for i in range(1, features.shape[3] + 1):
        if not single_visual:
            plt.subplot(4, 3, i)
            plt.xticks([])
            plt.yticks([])
        plt.imshow(features[0, :, :, i - 1], cmap="gray")
        if single_visual:
            plt.show()
            plt.figure(figsize=(16, 9))
    if not single_visual:
        plt.show()


def print_prediction(model, x_test, y_test):
    """
    Predict on images and print out actual vs predicted values.
    """
    for i in range(len(x_test)):
        single_x = img_to_array(load_img(x_test[i], color_mode="grayscale")) / 255
        # normally our datashape is (num_pictures, width, height, depth)
        # only one image will be (width, height, depth)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        print(single_x.shape)
        print(model.predict(single_x))
        print('actual', y_test.iloc[i])


def confusion_model(model, x_test, y_test):
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


def model_accuracy(model, x_test, y_test):
    """
    Print out model accuracy. Accuracy is not a good way of validating this model, it's here for class example.
    """
    memory_batch = 256
    validation_rounds = 0
    accumulative_accuracy = 0

    while len(x_test) > 0:
        current_x_train = []
        current_y_train = []
        for _ in range(memory_batch):
            if len(x_test) == 0:
                break
            if config.grayscale:
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


def get_image_index():
    """
    for visualizing images there are a few locations throughout the map that represent different cases
    """
    # All data from testing dataset session_2023_11_25-22_00_07
    # Turn 2 exit. Has white line, tire mark, inner wall tapering off
    index = 1688
    # Entering turn 2. White line, inner wall starting
    # index = 676
    # Middle turn 2
    # index = 818
    # middle of start strait
    # index = 212

    return index


if __name__ == "__main__":
    # Turn on to run model.validate on testing data.
    validate = False

    # Testing out different CNNs
    test_model = True

    image_index = get_image_index()

    recording_directory = config.linux_testing_directory

    if test_model:
        x, y = get_files(recording_directory)
        input_shape = img_to_array(load_img(x[0], color_mode=config.color_mode)).shape
        # keras_model = build_models.build_test_cnn(input_shape)
        keras_model = build_models.build_test_cnn_with_pooling(input_shape)
    else:
        keras_model, x, y = load_keras_model()

    # keras_model.summary()
    # exit()

    # conf_model(keras_model, x, y)
    # visualize_prediction(model, x, y)

    # visualize_kernels(model=keras_model)
    visualize_feature_map(keras_model, x)
    # visualize_model(keras_model, x)

    if validate:
        model_accuracy(keras_model, list(x), y.values.tolist())
