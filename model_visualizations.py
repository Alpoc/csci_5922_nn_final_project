"""
This module contains various visualizations to try and get a glimpse into what the model is learning.
"""
from utils import load_keras_model
from matplotlib import pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


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
    # for layer in model.layers:
    #     print(layer)
    for layer in model.layers:
        if "conv" not in layer.name:
            continue
        print(layer.name)
        feature_model = Model(inputs=model.inputs, outputs=layer.output)
        # plt.imshow(load_img(x_test[0]))
        single_x = img_to_array(load_img(x_test[0], color_mode="grayscale"))
        # plt.imshow(single_x)

        # normally our datashape is (num_pictures, width, height, depth)
        # only one image will be (width, height, depth)
        single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])
        features = feature_model.predict(single_x, verbose=False)

        fig = plt.figure(figsize=(16, 9))
        for i in range(1, features.shape[3] + 1):
            plt.subplot(8, 8, i)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(features[0, :, :, i-1], cmap="gray")

        plt.show()


if __name__ == "__main__":
    keras_model, x, y = load_keras_model()

    # visualize_kernels(model=keras_model)
    visualize_feature_map(keras_model, x)
