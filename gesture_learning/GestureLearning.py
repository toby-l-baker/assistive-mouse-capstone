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


IP = 'localhost'    # IP address for UDP connection
PORT = 5000         # port number for UDP connection
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
            gesture = np.argmax(model.predict(features))
        else:
            gesture = model.predict(features)[0]

        # display gesture
        if gesture == 0:
            print('CLOSED')
        elif gesture == 1:
            print('OK')
        elif gesture == 2:
            print('OPEN')
        else:
            print('UNKNOWN')


if __name__ == "__main__":
    main(sys.argv)
