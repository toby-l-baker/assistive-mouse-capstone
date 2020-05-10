#!/usr/bin/env python3

"""
Description: Gesture recognition via MediaPipe keypoints
Author: Ayusman Saha (aysaha@berkeley.edu)
"""
import os
import argparse
import socket

import numpy as np
import joblib
from keras import models

import keypoints as kp

IP = 'localhost'    # IP address of MediaPipe UDP client
PORT = 2000         # port number of MediaPipe UDP client
MAX_BYTES = 1024    # maximum number of bytes to read over UDP

def green(string):
    return "\033[92m{}\033[00m".format(string)

def yellow(string):
    return "\033[93m{}\033[00m".format(string)

def red(string):
    return "\033[91m{}\033[00m".format(string)
            
def display(gesture):
    if gesture == 0:
        # CLOSE
        string = green("CLOSE") + " | OPEN | FINGERS_ONE | FINGERS_TWO | THUMB_BENT"
    elif gesture == 1:
        # OPEN
        string = "CLOSE | " + green("OPEN") + " | FINGERS_ONE | FINGERS_TWO | THUMB_BENT"
    elif gesture == 2:
        # FINGERS_ONE
        string = "CLOSE | OPEN | " + green("FINGERS_ONE") + " | FINGERS_TWO | THUMB_BENT"
    elif gesture == 3:
        # FINGERS_TWO
        string = "CLOSE | OPEN | FINGERS_ONE | " + green("FINGERS_TWO") + " | THUMB_BENT"
    elif gesture == 4:
        # THUMB_BENT
        string = "CLOSE | OPEN | FINGERS_ONE | FINGERS_TWO | " + green("THUMB_BENT")
    else:
        # UNKNOWN
        string = red("CLOSE") + " | " + red("OPEN") + " | " + red("FINGERS_ONE") + " | " + red("FINGERS_TWO") + " | " + red("THUMB_BENT")
    
    print(string, end='\r')

def main(args):
    assert os.path.splitext(args.model)[1] == '.h5' or os.path.splitext(args.model)[1] == '.sav' 

    if args.model.endswith('.h5'):
        keras_model = True

        # load model
        model = models.load_model(args.model)

        # load data normalization parameters
        try:
            data = np.load(args.model.replace('.h5', '.npz'))
            mean = data['mean']
            std = data['std']
        except:
            print("Error: missing data normalization parameters")
            exit()
    elif args.model.endswith('.sav'):
        keras_model = False

        # load model
        model = joblib.load(args.model)

    print("=================================================================")
    print("Using model {}".format(args.model))
    print("-----------------------------------------------------------------")

    # establish UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))

    while True:
        try:
            # receive keypoints over UDP
            data, addr = sock.recvfrom(MAX_BYTES)
            keypoints = kp.decode(data)

            # convert keypoints to features
            features = kp.keypointsToFeatures(keypoints[:, :-1].flatten())
            features = features.reshape((1, -1))

            # skip bad data
            if np.isnan(np.sum(features)):
                continue

            # predict gesture
            if keras_model is True:
                features = kp.dataset.normalize(features, mean, std)
                output = model.predict(features)[0]
                gesture = np.argmax(output)
            else:
                gesture = model.predict(features)[0]
            
            # display gesture
            display(gesture)
        except KeyboardInterrupt:
            print("")
            break;

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('model')
    args = parser.parse_args()
    main(args)
