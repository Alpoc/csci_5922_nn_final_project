import pygame
import keyboard
import time
import csv
from os import path
from pygame.locals import JOYBUTTONDOWN

def setup_controllers(only_xbox=True):
    for i in range(0, pygame.joystick.get_count()):
        if only_xbox:
            if "Xbox One" in pygame.joystick.Joystick(i).get_name():
                controllers.append(pygame.joystick.Joystick(i))
                controllers[-1].init()
                print("Detected joystick ", controllers[-1].get_name(), "'")
    if len(controllers) > 1:
        while True:
            for controller in controllers:
                print(controller.get_button(1))
                if controller.get_button(1):
                    gamepad = controller
                    break
    print("connected count:", len(controllers))


def start_recording():
    print('starting recording')
    inputs = []
    while continue_recording:
        clock.tick(15)

        left_joystick_x_axis = controllers[0].get_axis(0)
        left_joystick_y_axis = controllers[0].get_axis(1)
        left_trigger = controllers[0].get_axis(4)
        right_trigger = controllers[0].get_axis(5)

        inputs.append([left_trigger, right_trigger, left_joystick_x_axis, left_joystick_y_axis])

        if keyboard.is_pressed(stop_recording_button):
            break

    time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
    recording_file = "controller_recording_" + time_str
    with open(path.join("controller_recordings", recording_file), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(inputs)
    print('recording ended')


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    continue_recording = True
    controllers = []

    start_recording_button = "k"
    stop_recording_button = "l"

    setup_controllers()

    while True:
        # if keyboard.is_pressed(start_recording_button):
        #     start_recording()
        #     break
        clock.tick(15)
        event = pygame.event.wait()
        # for event in pygame.event.get():
        if event.type == JOYBUTTONDOWN:
            if event.dict['button'] == 5:
                print("RB button hit")
                start_recording()
                break

    print("recording finished")
