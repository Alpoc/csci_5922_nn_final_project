import test_models
import utils
import config
import cv2

from tensorflow.keras.preprocessing.image import img_to_array, load_img


def crop_image(frame):
    x_start = 475
    x_end = 575

    y_start = 1075
    y_end = 1175

    frame = frame[x_start:x_end, y_start:y_end, :]

    print(frame.shape)
    cv2.imshow("image", frame)
    cv2.waitKey(0)


if __name__ == '__main__':

    recording_directory = config.linux_testing_directory
    x, y = utils.get_files(recording_directory)
    input_shape = img_to_array(load_img(x[0], color_mode=config.color_mode)).shape

    speed_model = test_models.build_speed_reader(input_shape=input_shape)

    single_x = img_to_array(load_img(x[0], color_mode="grayscale")) / 255
    # single_x = single_x.reshape(-1, single_x.shape[0], single_x.shape[1], single_x.shape[2])

    print(single_x.shape)

    crop_image(single_x)
