#!/usr/bin/env python3

import numpy as np
import cv2
import os
import sys


HEIGHT = 720
WIDTH = 720

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 24
THICKNESS = 24

TIMEOUT = 10
ENTER_KEY = 13

BASE_IMAGE = None
TEST_IMAGE = None
FULL_IMAGE = None
DRAWING = False


def callback(event, x, y, flags, param):
    global TEST_IMAGE, FULL_IMAGE, DRAWING
    
    # state machine to use mouse as a paintbrush
    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
    elif event == cv2.EVENT_LBUTTONUP:
        DRAWING = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if DRAWING is True:
            cv2.circle(TEST_IMAGE, (x, y), THICKNESS//2, (0, 0, 255), -1)
            cv2.circle(FULL_IMAGE, (x, y), THICKNESS//2, (0, 0, 255), -1)


def test(letter):
    global BASE_IMAGE, TEST_IMAGE, FULL_IMAGE

    # create images
    BASE_IMAGE = np.full((HEIGHT, WIDTH, 3), 255, dtype='uint8')
    TEST_IMAGE = np.full((HEIGHT, WIDTH, 3), 255, dtype='uint8')
    FULL_IMAGE = np.full((HEIGHT, WIDTH, 3), 255, dtype='uint8')

    # draw letter on image
    size = cv2.getTextSize(letter, FONT, SCALE, THICKNESS)
    cv2.putText(BASE_IMAGE, letter, (WIDTH//2 - size[1], HEIGHT//2 + size[1]), FONT, SCALE, (0, 0, 0), THICKNESS)
    cv2.putText(FULL_IMAGE, letter, (WIDTH//2 - size[1], HEIGHT//2 + size[1]), FONT, SCALE, (0, 0, 0), THICKNESS)

    # save base image to file
    cv2.imwrite('base.png', BASE_IMAGE)

    while True:
        # update image
        cv2.imshow('image', FULL_IMAGE)

        # wait for keyboard input
        key = cv2.waitKey(TIMEOUT)
        
        # finish if Enter key is pressed
        if key == ENTER_KEY:
            break
    
    # save test image to file
    cv2.imwrite('test.png', TEST_IMAGE)

    # load images as grayscale
    base_image = cv2.imread('base.png', cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

    # threshold test image and subtract from base_image
    thresh, thresh_image = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)

    # invert images
    base_image_inv = cv2.bitwise_not(base_image)
    thresh_image_inv = cv2.bitwise_not(thresh_image)

    # get overlapping region of the two images
    inside_image = cv2.bitwise_and(thresh_image_inv, base_image_inv)
    outside_image = cv2.bitwise_and(thresh_image_inv, base_image)

    # calculate number of pixels
    letter_pixels = int(np.sum(base_image_inv) / 255)   # number of pixels in the letter
    inside_pixels = int(np.sum(inside_image) / 255)     # number of pixels drawn inside the letter
    outside_pixels = int(np.sum(outside_image) / 255)   # number of pixels drawn outside the letter
    total_pixels = int(np.sum(thresh_image_inv) / 255)  # number of pixels drawn in total

    # calculate score
    if total_pixels == 0:
        score = 0
    else:
        score = 50 * (inside_pixels / letter_pixels) + 50 * (inside_pixels / total_pixels)

    # delete saved images
    os.remove('base.png')
    os.remove('test.png')

    return score


def main(args):
    scores = []

    # check for correct number of arguments
    if len(args) != 2:
        print('Usage: AccuracyTest.py test_string')
        exit()
    else:
        letters = args[1]

    # create window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', callback)

    # run a test for each letter
    for letter in letters:
        score = test(letter)
        scores.append(score)

    # delete window
    cv2.destroyWindow('image')
    
    # calculate average score
    average = sum(scores) / len(scores)

    # display scores
    for letter, score in zip(letters, scores):
        print(letter + ": {0:.2f}".format(score) + "%")

    print("average: {0:.2f}".format(average) + "%")


main(sys.argv)
