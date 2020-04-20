#!/usr/bin/env python3

"""
Description: Template for gesture recognition via machine learning
Author: Ayusman Saha
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import keypoints as kp

# --------------------------------------------------------------------------------------------------

SPLIT = 0.75                # split percentage for training vs. testing data
NORMALIZATION = 'cartesian' # type of data normalization ('cartesian' or 'polar')

# --------------------------------------------------------------------------------------------------

# NOTE: program needs keypoints.py which is located in gesture_learning/
def main(args):
    # check for correct arguments
    if len(args) != 2:
        # NOTE: data is located in gesture_learning/data/
        print("Usage: python template.py data") 
        exit()

    # process file
    with open(args[1], 'r') as f:
        train, test = kp.parse(f, shuffle=True, normalization=NORMALIZATION, split=SPLIT)

    # NOTE: training on a normal distribution can be easier for some approaches
    train.data = kp.dataset.normalize(train.data, train.mean, train.std)

    # NOTE: need to use training data information to normalize testing data
    test.data = kp.dataset.normalize(test.data, train.mean, train.std)

    '''
    do all machine learning work here

    train.data contains entries that are formatted as 20 (x,y) points in order. These points
    were generated from MediaPipe and correspond to keypoints on the user's hand. Although
    MediaPipe generates 21 points, the base of the hand is used as the origin (0,0) when
    normalizing and is therfore unnecessary.

    train.labels contains integers corresponding to different gestures. Each data entry has a
    corresponding label arranged such that train.data[i] is categorized by train.labels[i].
    Currently, the gesture classes for the 'fourClass' dataset are:
        0 - CLOSE
        1 - OK
        2 - OPEN
        3 - CLICK

    test.data is formatted the same as train.data and can be used to
    test the model against data it has never seen before

    test.labels is formatted the same as train.labels and can be used to
    quantify the accuracy of the model
    '''
    print("shape of training data: " + str(train.data.shape))
    print("shape of training labels: " + str(train.labels.shape))
    print("")
    print("shape of testing data: " + str(test.data.shape))
    print("shape of testing labels: " + str(test.labels.shape))

    # NOTE: save models in gesture_learning/models/

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main(sys.argv)
