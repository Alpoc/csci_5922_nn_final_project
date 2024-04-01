import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


import build_models
from utils import get_files

import numpy as np
import os
import gc
import config
import time


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
    # num epochs to run on current bach of data
    epochs = 1
    loop_count = 0
    start_time = epoch_time = time.time()
    batches_processed = 0

    epoch_run_times = []

    full_x_train = x_train.copy()
    full_y_train = y_train.copy()

    epochs_file = os.path.join(config.linux_model_location, "current_model", "epochs_ran.txt")
    if os.path.exists(epochs_file):
        with open(epochs_file, "r") as f:
            epochs_ran = int(f.read())
    else:
        epochs_ran = 0

    for _ in range(1000):
        while len(x_train) > 0:
            batch_start_time = time.time()
            current_x_train = []
            current_y_train = []
            print("loading images into memory")
            # This really takes a long time. There is a concurrency branch on github but it doesn't work reliably.
            for _ in range(config.memory_batch):
                if len(x_train) == 0:
                    break
                current_x_train.append(img_to_array(load_img(x_train.pop(0), color_mode=config.color_mode)) / 255)
                current_y_train.append(y_train.pop(0))

            # current_x_train, current_y_train = tf.keras.utils.timeseries_dataset_from_array(current_x_train, current_y_train, sequence_length=gpu_batch)

            # Datatypes must be np arrays. same as tf.stack().
            current_x_train = np.array(current_x_train)
            # current_x_train = tf.expand_dims(current_x_train, axis=0)
            current_y_train = np.array(current_y_train)
            print("Fitting batch in GPU")

            with tf.device("/GPU:0"):
                if save_callbacks:
                    model_hist = model.fit(current_x_train, current_y_train,
                                           epochs=epochs, batch_size=config.gpu_batch,
                                           callbacks=[cp_callback])
                else:
                    model_hist = model.fit(current_x_train, current_y_train,
                                           epochs=epochs, batch_size=config.gpu_batch)


            # python garbage collection was not working fast enough.
            del current_x_train
            del current_y_train
            # Todo: look into tracemalloc. gc.collect works but it's not perfect.
            gc.collect()

            # Intermediary info
            if loop_count >= 8:
                current_time = time.time()
                print(f'current runtime: {(current_time - start_time) / 60} minutes')
                # TF reports epoch time, but it takes a while to load the data into Memory
                batches_remaining = round(len(x_train) / config.memory_batch)
                avg_batch_time = (current_time - start_time) / batches_processed / 60
                print(f"estimated time remaining: {batches_remaining * avg_batch_time} minutes")
                print(f"epochs_ran: {epochs_ran}")
                # print('saving off model')
                # save_model(model)
                loop_count = 0
                memory_stats = tf.config.experimental.get_memory_info("GPU:0")
                peak_usage = round(memory_stats["peak"] / (2 ** 30), 3)
                print(f"peak memory usage: {peak_usage} GB.")
            loop_count += 1

            batches_processed += 1
            print(f"batch time: {time.time() - batch_start_time}")
            print(f'{round(len(x_train) / config.memory_batch)} batches to go!')

        print('saving model')
        save_model(model)
        epochs_ran += 1
        with open(epochs_file, "w") as f:
            f.write(str(epochs_ran))

        print(f"epochs_ran: {epochs_ran}")
        current_time = time.time()
        epoch_run_time = (current_time - epoch_time) / 60
        print(f'epoch time: {epoch_run_time} minutes')
        epoch_run_times.append(epoch_run_time)
        average_epoch_time = sum(epoch_run_times) / len(epoch_run_times)
        print(f'Average epoch run time: {average_epoch_time}')
        epoch_time = current_time

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
    """
    Simple method to check if the GPU is available
    """
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    # print("Num GPUs Available: ", num_gpus)
    # print(tf.test.is_built_with_cuda())
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
    # Getting a memory malloc error, trying to solve.
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    gpu_check()
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
        print("training model from scratch")
        move_previous_model_folder()
        input_shape = img_to_array(load_img(x[0], color_mode=config.color_mode)).shape
        if lstm_and_cnn_model:
            keras_model = build_models.build_cnn_lstm_model(input_shape)
        elif cnn:
            keras_model = build_models.build_cnn_model(input_shape)
        elif pure_lstm:
            keras_model = build_models.build_conv_lstm(input_shape)
        else:
            # Set whatever model you want manually
            keras_model = build_models.build_cnn_model_48x10_nn(input_shape)

    else:
        print("Continuing to train previous model")
        model_location = os.path.join(config.linux_model_location, "current_model", "keras_model_dir")
        keras_model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    keras_model.summary()
    train_in_batches(list(x), y.values.tolist(), keras_model)
