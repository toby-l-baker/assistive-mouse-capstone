#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 weihao <weihao@weihao-G7>
#
# Distributed under terms of the MIT license.
from sys import argv
import random
import time
from tkinter import *

from tkinter import messagebox

if len(argv) != 4:
    print('python3 button.py width height seed')
    exit()
script, width, height, seed = argv
height = int(height)
width = int(width)
random.seed(int(seed))

window_width = 800
window_height = 600
root = Tk()
root.title("button test")
root.geometry("%dx%d" % (window_width, window_height))

max_count = 10
count = 0
prev_button = None
def randomButton():
    global count
    global max_count
    global prev_button
    if prev_button != None:
        prev_button.destroy()

    if count >= max_count:
        print(time.time() - start, "seconds")
        root.quit()
    B = Button(root,  command = randomButton, bg="red")
    prev_button = B
    x = random.randint(0, window_width - width)
    y = random.randint(0, window_height - height)
    B.place(x = x,y = y, width=width, height=height)
    count += 1

start = time.time()
randomButton()
root.mainloop()
