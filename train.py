import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding,
                                     LSTM, ConvLSTM2D, TimeDistributed, BatchNormalization, Conv3D)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import os
import gc
import config
import time


def get_files(recording_base_dir):
    """
    TODO: use glob instead for easier code comprehension
    Cannot load the image files into memory. Too large.
    :return: list of files, json keypress file
    """
    frame_dir_list = []
    df_combined = pd.DataFrame()

    recordings = os.listdir(recording_base_dir)
    for session_dir in recordings:
        recording_dir = os.path.join(recording_base_dir, session_dir)

        video_frame_dir = os.path.join(recording_dir, "video_images")
        frame_list = os.listdir(video_frame_dir)

        frame_dir_list.extend([os.path.join(video_frame_dir, frame) for frame in frame_list])

        recording_file = 'inputs.csv'
        df = pd.read_csv(os.path.join(recording_dir, recording_file))
        df = df.drop(columns=df.columns[0], axis=1)
        df_combined = pd.concat([df_combined, df])

    return frame_dir_list, df_combined


def build_conv_lstm(input_shape):
    print((gpu_batch, *input_shape))
    new_model = Sequential([
        ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=False,
            activation="relu",
            input_shape=(gpu_batch, *input_shape)
        ),
        BatchNormalization(),
        # ConvLSTM2D(
        #     filters=64,
        #     kernel_size=(3, 3),
        #     padding="same",
        #     return_sequences=False,
        #     activation="relu",
        # ),
        # BatchNormalization(),
        # ConvLSTM2D(
        #     filters=64,
        #     kernel_size=(1, 1),
        #     padding="same",
        #     return_sequences=False,
        #     activation="relu",
        # ),
        Dense(4, activation='sigmoid'),
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    return new_model


def build_cnn_model(input_shape=None):
    """
    Build new model based on input image.
    :return: TensorFlow Keras model
    """
    if not input_shape:
        sample_x = img_to_array(load_img(x[0]))
        input_shape = sample_x.shape
    new_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return new_model


def build_cnn_lstm_model(input_shape):
    """
        Build new model based on input image.
        :return: TensorFlow Keras model
    """
    if not input_shape:
        sample_x = img_to_array(load_img(x[0]))
        input_shape = sample_x.shape
    # new_model = Sequential([
    #     TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='tanh'),
    #                     input_shape=(gpu_batch, 720, 1280, 1)),
    #     # ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', input_shape=(720, 1280, 1)),
    #     # MaxPooling2D(pool_size=(2, 2)),
    #     # ConvLSTM2D(filters=32, kernel_size=(4, 4), activation='tanh'),
    #     # MaxPooling2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dense(32, activation='relu'),
    #     # 4 labels.
    #     Dense(4, activation='sigmoid')
    # ])

    new_model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(32),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='sigmoid')
    ])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


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

    for _ in range(1000):
        while len(x_train) > 0:
            current_x_train = []
            current_y_train = []
            print("loading images into memory")
            for _ in range(memory_batch):
                if len(x_train) == 0:
                    break
                current_x_train.append(img_to_array(load_img(x_train.pop(0), color_mode=color_mode)) / 255)
                current_y_train.append(y_train.pop(0))

            # current_x_train, current_y_train = tf.keras.utils.timeseries_dataset_from_array(current_x_train, current_y_train, sequence_length=gpu_batch)

            # Datatypes must be np arrays. same as tf.stack().
            current_x_train = np.array(current_x_train)
            # current_x_train = tf.expand_dims(current_x_train, axis=0)
            current_y_train = np.array(current_y_train)
            print("moving images from RAM to VRAM")
            if len(tf.config.list_physical_devices('GPU')):
                with tf.device("/GPU:0"):
                    if save_callbacks:
                        model_hist = model.fit(current_x_train, current_y_train,
                                               epochs=epochs, batch_size=gpu_batch,
                                               callbacks=[cp_callback])
                    else:
                        model_hist = model.fit(current_x_train, current_y_train,
                                               epochs=epochs, batch_size=gpu_batch)

            # python garbage collection was not working fast enough.
            del current_x_train
            del current_y_train
            # Todo: look into tracemalloc. gc.collect works but it's not perfect.
            gc.collect()

            if loop_count >= 32:
                print('saving off model')
                current_time = time.time()
                print(f'current runtime: {(current_time - start_time) / 60} minutes')
                # TF reports epoch time, but it takes a while to load the data into Memory
                batches_remaining = round(len(x_train) / memory_batch)
                avg_batch_time = (current_time - start_time) / batches_processed / 60
                print(f"estimated time remaining: {batches_remaining * avg_batch_time} minutes")
                print(f"epochs_ran: {epochs_ran}")
                save_model(model)
                loop_count = 0
            loop_count += 1

            batches_processed += 1
            print(f'{round(len(x_train) / memory_batch)} batches to go!')

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
    color_mode = "grayscale"
    # type of NEW model to build. choose one. If not new model arch will be loaded of existing
    lstm_and_cnn_model = False
    cnn = False
    pure_lstm = True

    # Have to find a good balance for system memory and VRAM. gpu_batch will need to be changed depending on model size.
    # 32 batch exceeds 24GB VRAM.
    gpu_batch = 32
    # Load num images from storage to memory.
    memory_batch = 256

    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False
    recording_directory = config.linux_training_directory

    x, y = get_files(recording_directory)

    if train_new_model:
        move_previous_model_folder()
        input_shape = img_to_array(load_img(x[0], color_mode=color_mode)).shape
        if lstm_and_cnn_model:
            model = build_cnn_lstm_model(input_shape)
        elif cnn:
            model = build_cnn_model(input_shape)
        elif pure_lstm:
            model = build_conv_lstm(input_shape)

    else:
        model_location = os.path.join(config.linux_model_location, "current_model", "keras_model_dir")
        model = tf.keras.saving.load_model(model_location, custom_objects=None, compile=True, safe_mode=True)
    train_in_batches(list(x), y.values.tolist(), model)
