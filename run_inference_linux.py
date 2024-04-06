import time
import pygame
import cv2

from utils import load_keras_model

# Lots off incompatible libraries on windows vs linux
ON_WINDOWS = False

if ON_WINDOWS:
    import keyboard
    import dxcam
else:
    from vidgear.gears import ScreenGear
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

import config

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
        if False:
            # show screen for test
            if ON_WINDOWS:
                frame = camera.get_latest_frame()
            else:
                frame = stream.read()
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imshow("image", gray_image)
            cv2.waitKey(0)

        if True:
            if ON_WINDOWS:
                frame = camera.get_latest_frame()
            else:
                frame = stream.read()
            if config.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255
                frame = frame.reshape(-1, frame.shape[0], frame.shape[1])
            else:
                frame = frame.reshape(-1, frame.shape[0], frame.shape[1], frame.shape[2])
            # start_time = time.time()
            predicted_keys = model.predict(frame, verbose=0)
            # print(f'inference time {time.time() - start_time}')
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
        clock.tick_busy_loop(framerate)


if __name__ == "__main__":
    # run two inference steps and average the results together.
    average_results = False
    framerate = 60

    pygame.init()
    clock = pygame.time.Clock()

    keras_model, x, y = load_keras_model()

    inference_on_game(keras_model)

