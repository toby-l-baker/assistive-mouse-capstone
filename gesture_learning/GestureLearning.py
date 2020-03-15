#!/usr/bin/env python3

"""
Description: Gesture recognition via MediaPipe keypoints
Author: Ayusman Saha
"""
import sys
import socket
import numpy as np
import joblib
from keras import models
from Unsupervised import generateTrain, keypointsToFeatures
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
            
def display(gesture, output, supervised, valid=True):
    if supervised is True:
        if valid is True:
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
                string = "OPEN ({2:.4f}) | {3} ({0:.4f}) | OK ({1:.4f})".format(output[0], output[1], output[2], color("CLOSED"))
            elif gesture == 1:
                # OK
                string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | {3} ({1:.4f})".format(output[0], output[1], output[2], color("OK"))
            elif gesture == 2:
                # OPEN
                string = "{3} ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f})".format(output[0], output[1], output[2], color("OPEN"))
            else:
                # UNKNOWN
                string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f})".format(output[0], output[1], output[2])
        else:
            string = "OPEN ({2:.4f}) | CLOSED ({0:.4f}) | OK ({1:.4f})".format(0, 0, 0)
        
        print(string, end='\r')
    else:
        if valid is True:
            if gesture == 0:
                print("CLOSED")
            elif gesture == 1:
                print("OK")
            elif gesture == 2:
                print("OPEN")
            else:
                print("UNKNOWN")

def main(args):
    # check for correct arguments
    if not (2 <= len(args) <= 3):
        print("Usage: python GestureLearning.py model [data]")
        exit()
    
    if args[1].endswith('.h5'):
        supervised = True
        
        if len(args) != 3:
            print("Error: supervised model needs data for normalization")
            exit()

        # process file
        with open(args[2], 'r') as f:
            data, labels = generateTrain(f)
        
        # format data
        train = DataSet(data, labels)
        
        # load model
        model = models.load_model(args[1])
    elif args[1].endswith('.sav'):
        supervised = False

        # load model
        model = joblib.load(args[1])
    else:
        print("Error: model format not supported")
        exit()

    print("=====================================================")
    if supervised is True:
        print("Using supervised model [" + args[1] + "]")
    else:
        print("Using unsupervised model [" + args[1] + "]")
    print("-----------------------------------------------------")

    # establish UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))

    # initialize array for storing keypoints
    keypoints = np.zeros((21, 3))

    while True:
        # receive data
        data, addr = sock.recvfrom(MAX_BYTES)
        
        # parse data
        array = data.decode().strip(';').split(';')

        for index, value in enumerate(array):
            x, y, z = [float(i) for i in value.strip(',').split(',')]
            keypoints[index] = (x, y, z)

        # convert keypoints to features
        features = keypointsToFeatures(keypoints[:, :-1].flatten())
        features = features.reshape((1, features.shape[0]))
        
        # check if features are valid
        if np.isnan(np.sum(features)):
            display(-1, None, supervised, valid=False)
            continue
        
        if supervised is True:

            # predict gesture
            DataSet.normalize(features, train.mean, train.std)
            output = model.predict(features)[0]
            gesture = np.argmax(output)

            # display gesture
            display(gesture, output, supervised, valid=True)
        else:
            # predict gesture
            gesture = model.predict(features)[0]
            
            # display gesture
            display(gesture, None, supervised, valid=True)


if __name__ == "__main__":
    main(sys.argv)
