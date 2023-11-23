from pynput.keyboard import Key, Controller, Listener
import pygame
import time


def on_press(key):
    if key == Key.f9:
        print('starting playback')
        prev_time = time.time()
        with open('keylog.txt', 'r') as key_presses:
            for line in key_presses:
                key_string = line.split(',')[0].strip().replace("'", "")
                current_timestamp = line.split(',')[1].strip().replace("'", "")
                # curr_time = time.time()  # so now we have time after processing
                # diff = curr_time - prev_time  # frame took this much time to process and render
                # delay = max(1.0 / target_fps - diff,
                #             0)  # if we finished early, wait the remaining time to desired fps, else wait 0 ms!
                # print(delay)
                # # time.sleep(delay)
                # fps = 1.0 / (delay + diff)  # fps is based on total time ("processing" diff time + "wasted" delay time)
                # print(fps)
                # prev_time = curr_time

                if key_string == "Key.esc":
                    return False
                elif key_string == "Key.space":
                    keyboard.press(key.space)
                    time.sleep(delay)
                    keyboard.release(key.space)
                else:
                    keyboard.press(key_string)
                    time.sleep(delay)
                    keyboard.release(key_string)
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

