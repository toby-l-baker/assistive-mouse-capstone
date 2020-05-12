#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 weihao <weihao@weihao-G7>
#
# Distributed under terms of the MIT license.

"""

"""
import pyautogui
# NOTE set pyautogui.FAILSAFE = False is important   trust me im engineer
pyautogui.FAILSAFE = False
from collections import deque
import numpy as np
width, height = pyautogui.size()
left_down = False
right_down = False
skip_counter = 0
left_down_counter = 0
queue_size = 5
x_queue = deque()
y_queue = deque()
x_sum = 0
y_sum = 0

x_res_queue = deque()
y_res_queue = deque()
x_res_sum = 0
y_res_sum = 0

x_prev = y_prev = 0

# Mouse is a fifo where PipeWritingCalculator writes to, you can create it with 'mkfifo Mouse'
with open("Mouse") as f:
    for line in f:
        # skip 1 frame for every 2 frames since the frame reduction strategy employed by MediaPipe
        skip_counter += 1
        if skip_counter != 2:
            continue
        else:
            skip_counter = 0

        # the format of the input should be coordinate_x,coordinate_y,digit_representation_of_gesture
        split = line.rstrip('\n').split(",")
        x = float(split[0])
        y = float(split[1])
        gesture = int(split[2])

        # x_prev is the previous x coordinate, here we calculate the difference 
        x_diff = x - x_prev
        y_diff = y - y_prev
        x_prev = x
        y_prev = y


        # we use a queue to cache incoming coordinates, once the queue is full, we use the average to represent the x/y coordinate
        # basically a naive filter
        if len(x_queue) == queue_size:
            x_removed = x_queue.popleft()
            y_removed = y_queue.popleft()
            x_sum -= x_removed
            y_sum -= y_removed
            x_sum += x_diff
            y_sum += y_diff
            x_queue.append(x_diff)
            y_queue.append(y_diff)
        else:
            x_queue.append(x_diff)
            y_queue.append(y_diff)
            x_sum += x_diff
            y_sum += y_diff
            continue

        # when there is no hand
        if gesture == 0 and x_diff == 0 and y_diff == 0:
            left_down = False
            pyautogui.mouseUp()
        # when there is a hand
        else:
            # when slow down gesture detected, mediapipe +10 to the digit representation of gesture
            slow_down = gesture >= 10
            gesture = gesture % 10


            # gesture=4 means press left button
            # left_down_counter is used in the context of dragging. The usual scenario of a left click is you first slow down, then click
            # if you want to drag something, you also first slow down, then press left button
            # then you drag it to somewhere else. But you're still in slow down mode
            # well, but after 5 frames, if your gesture is still 4, the slow down mode will be cancelled.
            if gesture == 4:
                left_down_counter += 1
                if not left_down:
                    left_down = True
                    pyautogui.mouseDown()
            else:
                if left_down:
                    left_down = False
                    left_down_counter = 0
                    pyautogui.mouseUp()

            # gesture=2/3 means scroll up/down (maybe down/up)
            if gesture == 2:
                pyautogui.scroll(-1)
            if gesture == 3:
                pyautogui.scroll(1)

            # gesture=5 means right click
            if gesture == 5:
                if not right_down:
                    pyautogui.click(button='right')
                    right_down = True
            else:
                right_down = False

            # x_vel means x_velocity
            x_vel = x_diff
            y_vel = y_diff

            # some iir filter algorithm that I'm not satisfied with, feel free to implement your own
            if len(x_res_queue) != queue_size - 1:
                x_res_queue.append(x_diff)
                y_res_queue.append(y_diff)
                x_res_sum += x_diff
                y_res_sum += y_diff
                continue
            else:
                x_vel = (5 * x_sum - x_res_sum) / 21
                y_vel = (5 * y_sum - y_res_sum) / 21
                x_res_removed = x_res_queue.popleft()
                y_res_removed = y_res_queue.popleft()
                x_res_queue.append(x_vel)
                y_res_queue.append(y_vel)
                x_res_sum -= x_res_removed
                y_res_sum -= y_res_removed
                x_res_sum += x_vel
                y_res_sum += y_vel

            x_gain = np.tanh(abs(4*x_vel-2)) + 1
            y_gain = np.tanh(abs(4*y_vel-2)) + 1
            x_ = x_gain * x_vel * 3
            y_ = y_gain * y_vel * 3
            # the dragging thing that I mentioned earlier
            if left_down_counter > 5:
                slow_down = False
            # if slow down, the movement will be 10 times slower
            if slow_down:
                x_ *= 0.1
                y_ *= 0.1
            # the acutal move, NOTE set _pause=False is important, otherwise you'll find the script runs slower overtime
            pyautogui.moveRel(x_ * width, y_ * height, _pause=False)
