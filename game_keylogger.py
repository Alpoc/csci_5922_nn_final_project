import logging

import keyboard
import pygame
import time
from os import path
import csv

logging.basicConfig(filename="keylog.txt", level=logging.DEBUG, format="%(message)s")


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    record = False

    keystrokes = []
    timestep = 0

    print('starting')
    while True:
        clock.tick(30)

        if keyboard.is_pressed('esc'):
            break
        current_keys = []
        if keyboard.is_pressed('w'):
            current_keys.append('w')
        if keyboard.is_pressed('a'):
            current_keys.append('a')
        keystrokes.append([current_keys, time.time()])

    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    recording_file = "controller_recording_" + time_str
    with open(path.join("controller_recordings", recording_file), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(keystrokes)

    print("recording finished")
