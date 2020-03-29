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
from Unsupervised import keypointsToFeatures

IP = 'localhost'    # IP address of MediaPipe UDP client
PORT = 2000         # port number of MediaPipe UDP client
MAX_BYTES = 1024    # maximum number of bytes to read over UDP

def green(string):
    return "\033[92m{}\033[00m".format(string)

def yellow(string):
    return "\033[93m{}\033[00m".format(string)

def red(string):
    return "\033[91m{}\033[00m".format(string)
            
def display(gesture, output, keras_model=False):
    if keras_model is True:
        string = ""

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
            # CLOSE
            string += color("CLOSE") + " ({0:.4f}) | ".format(output[0])
            string += "OK ({0:.4f}) | ".format(output[1])
            string += "OPEN ({0:.4f}) | ".format(output[2])
            string += "CLICK ({0:.4f})".format(output[3])
        elif gesture == 1:
            # OK
            string += "CLOSE ({0:.4f}) | ".format(output[0])
            string += color("OK") + " ({0:.4f}) | ".format(output[1])
            string += "OPEN ({0:.4f}) | ".format(output[2])
            string += "CLICK ({0:.4f})".format(output[3])
        elif gesture == 2:
            # OPEN
            string += "CLOSE ({0:.4f}) | ".format(output[0])
            string += "OK ({0:.4f}) | ".format(output[1])
            string += color("OPEN") + " ({0:.4f}) | ".format(output[2])
            string += "CLICK ({0:.4f})".format(output[3])
        elif gesture == 3:
            # CLICK
            string += "CLOSE ({0:.4f}) | ".format(output[0])
            string += "OK ({0:.4f}) | ".format(output[1])
            string += "OPEN ({0:.4f}) | ".format(output[2])
            string += color("CLICK") + " ({0:.4f})".format(output[3])
        else:
            # UNKNOWN
            string += "CLOSE ({0:.4f}) | ".format(output[0])
            string += "OK ({0:.4f}) | ".format(output[1])
            string += "OPEN ({0:.4f}) | ".format(output[2])
            string += "CLICK ({0:.4f})".format(output[3])
        
        print(string, end='\r')
    else:
        string = ""
        color = green

        if gesture == 0:
            # CLOSE
            string = color("CLOSE") + " | OK | OPEN | CLICK"
        elif gesture == 1:
            # OK
            string = "CLOSE | " + color("OK") + " | OPEN | CLICK"
        elif gesture == 2:
            # OPEN
            string = "CLOSE | OK | " + color("OPEN") + " | CLICK"
        elif gesture == 3:
            # CLICK
            string = "CLOSE | OK | OPEN | CLICK" + color("CLICK")
        else:
            # UNKNOWN
            string = "CLOSE | OK | OPEN | CLICK"
        
        print(string, end='\r')

def main(args):
    # check for correct arguments
    if len(args) != 2:
        print("Usage: python GestureLearning.py model")
        exit()
    
    if args[1].endswith('.h5'):
        keras_model = True

        # load model
        model = models.load_model(args[1])

        # load data normalization parameters
        try:
            data = np.load(args[1].replace('.h5', '.npz'))
            mean = data['mean']
            std = data['std']
        except:
            print("Error: missing data normalization parameters")
            exit()
    elif args[1].endswith('.sav'):
        keras_model = False

        # load model
        model = joblib.load(args[1])
    else:
        print("Error: model format not supported")
        exit()

    print("=================================================================")
    print("Using model " + args[1])
    print("-----------------------------------------------------------------")

    # establish UDP connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))

    while True:
        # receive keypoints over UDP
        data, addr = sock.recvfrom(MAX_BYTES)
        keypoints = kp.decode(data)

        if keras_model is True:
            # normalize keypoints
            keypoints = kp.normalize_polar(keypoints)[1:, :-1].flatten()
            keypoints = kp.dataset.normalize(keypoints.reshape((1, keypoints.shape[0])), mean, std)

            # predict gesture
            output = model.predict(keypoints)[0]
            gesture = np.argmax(output)
        else:
            # convert keypoints to features
            features = keypointsToFeatures(keypoints[:, :-1].flatten())
            features.reshape((1, features.shape[0]))

            # predict gesture
            gesture = model.predict(features)[0]
            output = None

        # display gesture
        display(gesture, output, keras_model)

if __name__ == '__main__':
    main(sys.argv)
