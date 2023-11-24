import logging
import os

import keyboard
import pygame
from os import path
import json
import dxcam
import time
from PIL import Image


def create_recording_dir():
    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    recording_file = "session_" + time_str
    session_path = path.join('recordings', recording_file)
    os.mkdir(session_path)
    video_path = path.join(session_path, "video_images")
    os.mkdir(video_path)
    return session_path, video_path


def record_session():
    session_path, video_path = create_recording_dir()
    i = 0
    record = False

    print('ready to record')
    while True:

        if keyboard.is_pressed('f9'):
            record = True
            print('recording started')

        if record:
            if keyboard.is_pressed('esc') or keyboard.is_pressed('f9') and len(keystrokes) > 10:
                break
            current_keys = []
            if keyboard.is_pressed('w'):
                current_keys.append('w')
            if keyboard.is_pressed('a'):
                current_keys.append('a')
            if keyboard.is_pressed('s'):
                current_keys.append('s')
            if keyboard.is_pressed('d'):
                current_keys.append('d')

            keystrokes.append([current_keys, time.time()])
            # storing in memory fills up very quickly.
            # images.append(camera.get_latest_frame())

            img = Image.fromarray(camera.get_latest_frame()).convert("RGB")
            # tif format saves much faster than png
            img.save(path.join(video_path, str(i) + '.tif'))

            i += 1

        # limit the loop to specified "fps" value
        clock.tick_busy_loop(30)

    camera.stop()

    print("recording ended")
    print(f'keys captured: {len(keystrokes)}')

    with open(path.join(session_path, 'inputs' + '.json'), "w", newline='') as f:
        f.write(json.dumps(keystrokes, indent=4))


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    keystrokes = []
    images = []
    # https://github.com/ra1nty/DXcam
    camera = dxcam.create()
    camera.start(region=(0, 0, 1920, 1080), target_fps=30)

    record_session()

