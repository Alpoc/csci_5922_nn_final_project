import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, CategoryEncoding
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os



def get_files(recording_base_dir):
    """
    Cannot load the image files into memory. Too large.
    :return: list of files, json keypress file
    """
    video_frames = []

    # recording_base_dir = "recordings"
    recordings = os.listdir(recording_base_dir)
    for session_dir in recordings:
        recording_dir = os.path.join(recording_base_dir, session_dir)

        video_frame_dir = os.path.join(recording_dir, "video_images")
        frame_list = os.listdir(video_frame_dir)

        frame_dir_list = [os.path.join(video_frame_dir, frame) for frame in frame_list]

        recording_file = 'inputs.csv'
        df = pd.read_csv(os.path.join(recording_dir, recording_file))
        df = df.drop(columns=df.columns[0], axis=1)

    return frame_dir_list, df


def load_model_onto_gpu():

    # x = imageio.imread(x_train[0]))
    # sample_x = cv2.imread(x_train[0])
    sample_x = img_to_array(load_img(x_train[0]))
    gpu_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=sample_x.shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        # 4 labels.
        Dense(4, activation='softmax')
        # tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    ])
    gpu_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return gpu_model


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
    gpu_batch = 16
    memory_batch = 512

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
                model_hist = model.fit(current_x_train, np.array(current_y_train), epochs=8, batch_size=gpu_batch,
                                       callbacks=[cp_callback])
        print(f'remaining data: {len(x_train)}')


def test_gpu():
    print('yum')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda())

if __name__ == "__main__":
    # test_gpu()
    # exit()

    recording_directory = "/media/dj/Games Drive/Nerual_networks"

    x, y = get_files(recording_directory)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print(f"training set x: {len(x_train)}")
    print(f"training set y: {len(y_train)}")
    print(f"test set x: {len(x_test)}")
    print(f"test set y: {len(y_test)}")

    # Data Normalized to 0 to 1. Pixel values are 0-255
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # x_test = np.array([img_to_array(load_img(x)) for x in x_test])

    model = load_model_onto_gpu()
    train_in_batches(list(x_train), y_train.values.tolist(), model, x_test, y_test.to_numpy())