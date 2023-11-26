import logging
import os

import keyboard
import pygame
from os import path
import json
import dxcam
import time
from PIL import Image
from multiprocessing import Pool
import pandas as pd
import asyncio
import threading

def create_recording_dir():
    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    recording_file = "session_" + time_str
    if not path.exists('recordings'):
        os.mkdir("recordings")
    session_path = path.join('recordings', recording_file)
    os.mkdir(session_path)
    video_path = path.join(session_path, "video_images")
    os.mkdir(video_path)
    return session_path, video_path


def frame_writer(frame_dict):
    """
    Itterate through dictionary saving off images.
    May need to look into del dict entries to clear memory as we process. TBD
    :param frame_dict: key: save path, value: nparray image
    :return:
    """
    for path_string, frame in frame_dict.items():
        img = Image.fromarray(frame).convert("RGB")
        # tif format saves much faster than png
        img.save(path_string)

def record_session():
    session_path, video_path = create_recording_dir()
    i = 0
    record = False

    frames = {}

    print('ready to record')
    while True:
        if keyboard.is_pressed('f9'):
            record = True
            print('recording started')

        if record:
            if keyboard.is_pressed('esc') or keyboard.is_pressed('f9') and len(keystrokes) > 10:
                threading.Thread(target=frame_writer, args=(frames,)).start()
                frames = {}
                break
            current_keys = [0, 0, 0, 0]
            if keyboard.is_pressed('w'):
                current_keys[0] = 1
            if keyboard.is_pressed('a'):
                current_keys[1] = 1
            if keyboard.is_pressed('s'):
                current_keys[2] = 1
            if keyboard.is_pressed('d'):
                current_keys[3] = 1

            keystrokes.append(current_keys)
            # storing in memory fills up very quickly.
            # images.append(camera.get_latest_frame())

            if not thread:
                img = Image.fromarray(camera.get_latest_frame()).convert("RGB")
                # tif format saves much faster than png
                img.save(path.join(video_path, str(i).zfill(10) + '.tif'))
            else:
                # TBD if this actually improves frame capture FPS.
                frames[path.join(video_path, str(i).zfill(10) + '.tif')] = camera.get_latest_frame()
                if len(frames) > 60:
                    threading.Thread(target=frame_writer, args=(frames,)).start()
                    frames = {}
            i += 1

        # limit the loop to specified "fps" value
        clock.tick_busy_loop(target_framerate)

    camera.stop()

    print("recording ended")
    print(f'keys captured: {len(keystrokes)}')

    # with open(path.join(session_path, 'inputs' + '.json'), "w", newline='') as f:
    #     f.write(json.dumps(keystrokes, indent=4))
    df = pd.DataFrame(keystrokes, columns=['w', 'a', 's', 'd'])
    df.to_csv(path.join(session_path, 'inputs' + '.csv'))

if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    keystrokes = []
    images = []
    target_framerate = 30
    # https://github.com/ra1nty/DXcam
    camera = dxcam.create(device_idx=0, output_idx=1)
    camera.start(region=(0, 0, 1280, 720), target_fps=target_framerate)
    thread = True
    record_session()
