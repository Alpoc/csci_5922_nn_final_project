import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import gc



def get_files(recording_base_dir):
    """
    TODO: use glob instead for easier code comprehension
    Cannot load the image files into memory. Too large.
    :return: list of files, json keypress file
    """
    frame_dir_list = []
    df_combined = pd.DataFrame()

    # recording_base_dir = "recordings"
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


def build_model(input_shape=None):
    """
    Build new model based on input image.
    :return: TensorFlow Keras model
    """
    if not input_shape:
        sample_x = img_to_array(load_img(x_train[0]))
        input_shape = sample_x.shape
    new_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='softmax')
        # tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    ])
    new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return new_model


def train_in_batches(x_train, y_train, model, x_test, y_test):
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
    # Have to find a good balance for system memory and VRAM
    # 32 batch exceeds 24GB VRAM.
    gpu_batch = 16
    # System: 32GB ram, os uses 5GB
    # 512 is a good match for leaving 2GB free but when laoding in a new batch is OOMs for some reason.
    # 384 is about 5GB each batch but the initial load uses 10GB. Not sure why
    memory_batch = 384
    epochs = 8
    loop_count = 0

    while len(x_train) > 0:
        current_x_train = []
        current_y_train = []
        for _ in range(memory_batch):
            if len(x_train) == 0:
                break
            current_x_train.append(img_to_array(load_img(x_train.pop(0))) / 255)
            current_y_train.append(y_train.pop(0))
        current_x_train = np.array(current_x_train)

        if len(tf.config.list_physical_devices('GPU')):
            with tf.device("/GPU:0"):
                if save_callbacks:
                    model_hist = model.fit(current_x_train, np.array(current_y_train),
                                           epochs=epochs, batch_size=gpu_batch,
                                           callbacks=[cp_callback])
                else:
                    model_hist = model.fit(current_x_train, np.array(current_y_train),
                                           epochs=epochs, batch_size=gpu_batch)

        # python garbage collection was not working fast enough.
        del current_x_train
        del current_y_train
        # Todo: look into tracemalloc. gc.collect works but it's not perfect.
        gc.collect()

        if loop_count >= 5:
            tf.keras.saving.save_model(model, "models/current_model.keras", overwrite=True)
            loop_count = 0
        loop_count += 1

        print(f'remaining data: {len(x_train)}')

    tf.keras.saving.save_model(model, "models/current_model.keras", overwrite=True)


def gpu_check():
    print('yum')
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", num_gpus)
    print(tf.test.is_built_with_cuda())
    if not num_gpus:
        print('only cpu training available. Remove gpu check to continue')
        exit()

if __name__ == "__main__":
    # gpu_check()
    # callbacks are saved after each epoc. It's not great in our case since we're batching data into the RAM.
    save_callbacks = False
    recording_directory = "/media/dj/Games Drive/Nerual_networks"
    model_location = "/media/dj/Games Drive/Nerual_networks/models/current_model.keras"

    x, y = get_files(recording_directory)
    # for i in range(len(x)):
    #     print(f"x: {x[i]},   y{y.iloc[i]}")
    # exit()


    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print(f"training set x: {len(x_train)}")
    print(f"training set y: {len(y_train)}")
    print(f"test set x: {len(x_test)}")
    print(f"test set y: {len(y_test)}")

    # Data Normalized to 0 to 1. Pixel values are 0-255
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # x_test = np.array([img_to_array(load_img(x)) for x in x_test])

    model = build_model()
    train_in_batches(list(x_train), y_train.values.tolist(), model, x_test, y_test.to_numpy())