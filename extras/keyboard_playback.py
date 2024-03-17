"""
A small script for testing keyboard playback.
"""

from pynput.keyboard import Key, Controller, Listener
import pygame
import time
from os import path
import json

def on_press(key):
    if key == Key.f9:
        print('starting playback')
        prev_keys = []
        prev_time = 0
        # recording_file = "controller_recording_2023_11_23-12_32_47"
        recording_file = 'controller_recording_2023_11_24-10_29_13.csv'


        with open(path.join('controller_recordings', recording_file), 'r') as key_file:
            key_presses = json.load(key_file)
            for time_step in key_presses:
                # key_list = list(line.split(',')[0].strip().replace("'", ""))
                # current_timestamp = line.split(',')[1].strip().replace("'", "")

                for key_string in time_step[0]:
                    if key_string == "Key.f9" or "Key.esc":
                        return False
                    elif key_string == "Key.space":
                        keyboard.press(key.space)
                    else:
                        print(key_string, "down")
                        keyboard.press(key_string)

                    try:
                        prev_keys.remove(key_string)
                    except ValueError:
                        pass

                for key_string in prev_keys:
                    if key_string == "Key.esc":
                        return False
                    elif key_string == "Key.space":
                        keyboard.release(key.space)
                    else:
                        keyboard.release(key_string)
                        print(key_string, 'up')

                prev_keys = time_step[0]

                # curr_time = time.time()  # so now we have time after processing
                # diff = curr_time - prev_time  # frame took this much time to process and render
                # delay = max(1.0 / target_fps - diff,
                #             0)  # if we finished early, wait the remaining time to desired fps, else wait 0 ms!
                # print(delay)
                # # time.sleep(delay)
                # fps = 1.0 / (delay + diff)  # fps is based on total time ("processing" diff time + "wasted" delay time)
                # print(fps)
                # prev_time = curr_time
                # time.sleep(delay)



        print("playback complete")
        return False


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    record = False

    keyboard = Controller()

    dt = 60 / 1000
    target_fps = 30

    with Listener(on_press=on_press) as listener:
        listener.join()

