"""
Description: Functions to manipulate MediaPipe keypoints
Author: Ayusman Saha
"""
import numpy as np
import pandas as pd

NUM_KEYPOINTS = 21  # number of keypoints

# calculates the length of a vector
def length(vector):
    return np.linalg.norm(vector)

# calculates the angle of a vector with respect to the x-axis
def angle(vector):
    return np.arctan2(vector[1], vector[0])

# expresses a point in a different coordinate frame
def rigid_body_transform(point, origin, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    g = np.eye(3)
    g[0:2, 0:2] = R.transpose()
    g[0:2, 2] = (-1 * R.transpose()) @ origin
    return (g @ np.append(point, 1))[:-1]

# normalizes keypoints in cartesian coordinates
def normalize_cartesian(keypoints):
    normalized = np.zeros(keypoints.shape)

    if keypoints.shape == (NUM_KEYPOINTS, 3):
        origin = keypoints[0, :-1]
        fixed = keypoints[9, :-1]
        theta = angle(fixed - origin) + np.pi/2

        for index, keypoint in enumerate(keypoints):
            normalized[index, :-1] = rigid_body_transform(keypoint[:-1], origin, theta) * np.array([1, -1])
    elif keypoints.shape == (NUM_KEYPOINTS, 2):
        origin = keypoints[0]
        fixed = keypoints[9]
        theta = angle(fixed - origin) + np.pi/2

        for index, keypoint in enumerate(keypoints):
            normalized[index] = rigid_body_transform(keypoint, origin, theta) * np.array([1, -1])
    else:
        raise ValueError("keypoints array does not have shape (" + str(NUM_KEYPOINTS) + ",3) or (" + str(NUM_KEYPOINTS) + ",2)")

    return normalized

# normalizes keypoints in polar coordinates
def normalize_polar(keypoints):
    normalized = np.zeros(keypoints.shape)

    if keypoints.shape == (NUM_KEYPOINTS, 3):
        origin = keypoints[0, :-1]
        fixed = keypoints[9, :-1]
        theta = angle(fixed - origin) + np.pi/2

        for index, keypoint in enumerate(keypoints):
            vector = keypoint[:-1] - origin
            normalized[index, :-1] = np.array([length(vector), angle(vector) - theta]) * np.array([1, -1])
    elif keypoints.shape == (NUM_KEYPOINTS, 2):
        origin = keypoints[0]
        fixed = keypoints[9]
        theta = angle(fixed - origin) + np.pi/2

        for index, keypoint in enumerate(keypoints):
            vector = keypoint - origin
            normalized[index] = np.array([length(vector), angle(vector) - theta]) * np.array([1, -1])
    else:
        raise ValueError("keypoints array does not have shape (" + str(NUM_KEYPOINTS) + ",3) or (" + str(NUM_KEYPOINTS) + ",2)")

    return normalized

# encodes keypoints as data
def encode(keypoints):
    data = ""

    if keypoints.shape == (NUM_KEYPOINTS, 3):
        for index, keypoint in enumerate(keypoints):
            x, y, z = keypoint
            data += str(x) + ',' + str(y) + ',' + str(z) + ';'
    elif keypoints.shape == (NUM_KEYPOINTS, 2):
        for index, keypoint in enumerate(keypoints):
            x, y = keypoint
            z = 0.0
            data += str(x) + ',' + str(y) + ',' + str(z) + ';'
    else:
        raise ValueError("keypoints array does not have shape (" + str(NUM_KEYPOINTS) + ",3) or (" + str(NUM_KEYPOINTS) + ",2)")

    return data.encode()

# decodes keypoints from data
def decode(data):
    keypoints = np.zeros((NUM_KEYPOINTS, 3))
    array = data.decode().strip(';').split(';')
    
    for index, value in enumerate(array):
        x, y, z = [float(num) for num in value.strip(',').split(',')]
        keypoints[index] = (x, y, z)

    return keypoints

# parses file made up of lines containing 21 (x,y) keypoints and a label
def parse(f, normalization=None):
    raw = np.array(pd.read_csv(f, sep=',', header=None).values[1:])
    data = raw[:, :-1].astype('float64') 
    labels = raw[:, -1].astype('int8')

    if normalization == 'cartesian':
        for index, entry in enumerate(data):
            entry = entry.reshape(NUM_KEYPOINTS, entry.shape[0]//NUM_KEYPOINTS)
            data[index] = normalize_cartesian(entry).flatten()
    elif normalization == 'polar':
        for index, entry in enumerate(data):
            entry = entry.reshape(NUM_KEYPOINTS, entry.shape[0]//NUM_KEYPOINTS)
            data[index] = normalize_polar(entry).flatten()

    return data, labels

# displays keypoints
def display(keypoints):
    for index, keypoint in enumerate(keypoints):
        print("keypoint[" + str(index) + "] = " + str(keypoint))