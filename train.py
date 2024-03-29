import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import build_models
from build_models import build_cnn_model, build_cnn_lstm_model, build_conv_lstm
from utils import get_files

import numpy as np
import os
import gc
import config
import time
from threading import Thread
import asyncio

memory_x_batches = []
memory_y_batches = []


def load_images_into_memory(x_train, y_train):
    full_x_train = x_train.copy()
    full_y_train = y_train.copy()
    while True:
        if len(memory_x_batches) > 1:
            # print("batch filled")
            time.sleep(0.25)
            continue
        next_x_train = []
        next_y_train = []
        print("loading images into memory")
        for _ in range(config.memory_batch):
            if len(x_train) == 0:
                x_train = full_x_train.copy()
                y_train = full_y_train.copy()
                break
            next_x_train.append(img_to_array(load_img(x_train.pop(0), color_mode=config.color_mode)) / 255)
            next_y_train.append(y_train.pop(0))
        memory_x_batches.append(np.array(next_x_train))
        memory_y_batches.append(np.array(next_y_train))
        print("done loading into memory")

def train_in_batches(x_train, y_train, model):
    """
    We cannot load all of the frames into memory so do it incrementally.
    :param x_train:
    :param y_train:
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """



    checkpoint_path = "model_weights/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    epochs = 1
    loop_count = 0
    start_time = epoch_time = time.time()
    batches_processed = 0

    full_x_train = x_train.copy()
    full_y_train = y_train.copy()
    epochs_ran = 0

    # loop = asyncio.get_event_loop()
    # loop.create_task(load_images_into_memory(x_train, y_train))
    # loop.run_forever()
    thread = Thread(target=load_images_into_memory, args=(x_train, y_train))
    thread.start()


    for _ in range(5):
        # load_images_into_memory()
        while len(x_train) > 0:
            batch_start_time = time.time()

            # current_x_train, current_y_train = tf.keras.utils.timeseries_dataset_from_array(current_x_train, current_y_train, sequence_length=gpu_batch)

            while len(memory_x_batches) == 0:
                pass
            # Datatypes must be np arrays. same as tf.stack().
            current_x_train = memory_x_batches.pop()
            # current_x_train = tf.expand_dims(current_x_train, axis=0)
            current_y_train = memory_y_batches.pop()
            print("moving images from RAM to VRAM")
            if len(tf.config.list_physical_devices('GPU')):
                with tf.device("/GPU:0"):
                    if save_callbacks:
                        model_hist = model.fit(current_x_train, current_y_train,
                                               epochs=epochs, batch_size=config.gpu_batch,
                                               callbacks=[cp_callback])
                    else:
                        model_hist = model.fit(current_x_train, current_y_train,
                                               epochs=epochs, batch_size=config.gpu_batch)

            # python garbage collection was not working fast enough.
            # del current_x_train
            # del current_y_train
            # # Todo: look into tracemalloc. gc.collect works but it's not perfect.
            # gc.collect()

            if loop_count >= 32:
                print('saving off model')
                current_time = time.time()
                print(f'current runtime: {(current_time - start_time) / 60} minutes')
                # TF reports epoch time, but it takes a while to load the data into Memory
                batches_remaining = round(len(x_train) / config.memory_batch)
                avg_batch_time = (current_time - start_time) / batches_processed / 60
                print(f"estimated time remaining: {batches_remaining * avg_batch_time} seconds")
                print(f"epochs_ran: {epochs_ran}")
                save_model(model)
                loop_count = 0
            loop_count += 1

            batches_processed += 1
            print(f"batch time: {time.time() - batch_start_time}")
            print(f'{round(len(x_train) / config.memory_batch)} batches to go!')


        print('saving model')
        save_model(model)
        epochs_ran += 1
        current_time = time.time()
        print(f'epoch time: {current_time - epoch_time} seconds')
        epoch_time = current_time
        print('overwriting list')
        x_train = full_x_train.copy()
        y_train = full_y_train.copy()


def move_previous_model_folder():
    """
    Rename previous model to not overwrite it
    :return:
    """
    model_path = os.path.join(config.linux_model_location, "current_model")
    if os.path.exists(model_path):
        os.rename(model_path, os.path.join(config.linux_model_location, "model_" + time.strftime("%Y_%m_%d-%H_%M_%S")))
        os.mkdir(model_path)


def gpu_check():
    print('yum')
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", num_gpus)
    print(tf.test.is_built_with_cuda())
    if not num_gpus:
        print('only cpu training available. Remove gpu check to continue')
        exit()


def save_model(model):
    """
    Converting to run on windows is tricky. Save off in a few different approaches.
    :param model: keras model
    :return:
    """
    # Saves model in directory for windows.
    tf.keras.saving.save_model(model,
                               os.path.join(config.linux_model_location, "current_model", "keras_model_dir"),
                               overwrite=True)
    # save model in Keras native format.
    tf.keras.saving.save_model(model,
                               os.path.join(config.linux_model_location, "current_model", "my_model.keras"),
                               overwrite=True)


if __name__ == "__main__":
    # gpu_check()
    # It not train_new_model the "current_model" will be loaded and training will continue.
    train_new_model = False

    # type of NEW model to build. choose one. If not new model arch will be loaded of existing
    lstm_and_cnn_model = False
    cnn = False
    pure_lstm = False

    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False
    recording_directory = config.linux_training_directory

    x, y = get_files(recording_directory)

    if train_new_model:
        move_previous_model_folder()
        input_shape = img_to_array(load_img(x[0], color_mode=config.color_mode)).shape
        if lstm_and_cnn_model:
            keras_model = build_cnn_lstm_model(input_shape)
        elif cnn:
            keras_model = build_cnn_model(input_shape)
        elif pure_lstm:
            keras_model = build_conv_lstm(input_shape)
        else:
            keras_model = build_models.build_test_cnn_model(input_shape)

    else:
        model_location = os.path.join(config.linux_model_location, "current_model", "keras_model_dir")
        keras_model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    keras_model.summary()
    train_in_batches(list(x), y.values.tolist(), keras_model)
