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
            continue

        # predict gesture
        if supervised is True:
            DataSet.normalize(features, train.mean, train.std)
            output = model.predict(features)[0]
            gesture = np.argmax(output)
            confidence = " ({0:.2f}%)".format(output[gesture] * 100)
        else:
            gesture = model.predict(features)[0]
            confidence = ""

        # display gesture
        if gesture == 0:
            print("CLOSED" + confidence)
        elif gesture == 1:
            print("OK" + confidence)
        elif gesture == 2:
            print("OPEN" + confidence)
        else:
            print("UNKNOWN")


if __name__ == "__main__":
    main(sys.argv)
