#!/usr/bin/env python3

"""
Description: Gesture recognition via MediaPipe keypoints
Author: Ayusman Saha
"""
import sys
import socket
import numpy as np
import keypoints as kp
import joblib
from keras import models
from DLNN import DataSet

IP = 'localhost'    # IP address of MediaPipe UDP client
PORT = 2000         # port number of MediaPipe UDP client
MAX_BYTES = 1024    # maximum number of bytes to read over UDP

def green(string):
    return "\033[92m{}\033[00m".format(string)

def yellow(string):
    return "\033[93m{}\033[00m".format(string)

def red(string):
    return "\033[91m{}\033[00m".format(string)
            
def display(gesture, output, supervised):
    if supervised is True:
        if output[gesture] > 0.95:
            # high accuracy
            color = green
        elif output[gesture] > 0.75:
            # medium accuracy
            color = yellow
        else:
            # low accuracy
            color = red

        if gesture == 0:
            # CLOSED
            string = "OPEN ({2:.4f}) | {4} ({0:.4f}) | OK ({1:.4f}) | CLICK ({3:.4f})".format(output[0], output[1], output[2], output[3], color("CLOSED"))
        elif gesture == 1:
            # OK
            string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | {4} ({1:.4f}) | CLICK ({3:.4f})".format(output[0], output[1], output[2], output[3], color("OK"))
        elif gesture == 2:
            # OPEN
            string = "{4} ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f}) | CLICK ({3:.4f})".format(output[0], output[1], output[2], output[3], color("OPEN"))
        elif gesture == 3:
            # CLICK
            string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f}) | {4} ({3:.4f})".format(output[0], output[1], output[2], output[3], color("CLICK"))
        else:
            # UNKNOWN
            string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f}) | CLICK ({3:.4f})".format(output[0], output[1], output[2], output[3])
        
        print(string, end='\r')
    else:
        if gesture == 0:
            print("CLOSED")
        elif gesture == 1:
            print("OK")
        elif gesture == 2:
            print("OPEN")
        elif gesture == 3:
            print("CLICK")
        else:
            print("UNKNOWN")

def main(args):
    # check for correct arguments
    if len(args) != 2:
        print("Usage: python GestureLearning.py model")
        exit()
    
    if args[1].endswith('.h5'):
        supervised = True

        # load model
        model = models.load_model(args[1])

        # load data normalization parameters
        try:
            mean, std = np.load(args[1].replace('.h5', '.npy'))
        except:
            print("Error: missing data normalization parameters")
            exit()
    elif args[1].endswith('.sav'):
        supervised = False

        # load model
        model = joblib.load(args[1])
    else:
        print("Error: model format not supported")
        exit()

    print("==============================================================")
    if supervised is True:
        print("Using supervised model [" + args[1] + "]")
    else:
        print("Using unsupervised model [" + args[1] + "]")
    print("--------------------------------------------------------------")

    # establish UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))

    while True:
        # receive data
        data, addr = sock.recvfrom(MAX_BYTES)

        # processes data
        keypoints = kp.decode(data)
        keypoints = kp.normalize_cartesian(keypoints)[1:, :-1].flatten()
        keypoints = keypoints.reshape((1, keypoints.shape[0]))
        keypoints = DataSet.normalize(keypoints, mean, std)

        if supervised is True:
            # predict gesture
            output = model.predict(keypoints)[0]
            gesture = np.argmax(output)

            # display gesture
            display(gesture, output, supervised)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main(sys.argv)
