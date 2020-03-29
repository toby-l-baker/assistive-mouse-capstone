"""
Description: Functions to manipulate MediaPipe keypoints
Author: Ayusman Saha
"""
import numpy as np
import pandas as pd

NUM_KEYPOINTS = 21  # number of keypoints

# container for learning data and labels
class dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        if data.shape[0] != 0:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        else:
            self.mean = None
            self.std = None

    def normalize(data, mean, std):
        return (data - mean) / std

    def shuffle(a, b):
        p = np.random.permutation(min(len(a), len(b)))
        return a[p], b[p]

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
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape((NUM_KEYPOINTS, -1))
        flatten = True
    else:
        flatten = False

    normalized = np.zeros(keypoints.shape)
    origin = keypoints[0, 0:2]
    fixed = keypoints[9, 0:2]
    theta = angle(fixed - origin) + np.pi/2

    for index, keypoint in enumerate(keypoints):
        normalized[index, 0:2] = rigid_body_transform(keypoint[0:2], origin, theta) * np.array([1, -1])

    if flatten is True:
        normalized = normalized.flatten()

    return normalized

# normalizes keypoints in polar coordinates
def normalize_polar(keypoints):
    if keypoints.ndim == 1:
        keypoints = normalize_cartesian(keypoints.reshape((NUM_KEYPOINTS, -1)))
        flatten = True
    else:
        keypoints = normalize_cartesian(keypoints)
        flatten = False

    normalized = np.zeros(keypoints.shape)

    for index, keypoint in enumerate(keypoints):
        normalized[index, 0:2] = (length(keypoint[0:2]), angle(keypoint[0:2]))

    if flatten is True:
        normalized = normalized.flatten()

    return normalized

# converts keypoints to a stirng
def to_string(keypoints):
    string = ""

    if keypoints.ndim == 1:
        keypoints = keypoints.reshape((NUM_KEYPOINTS, -1))

    if keypoints.shape == (NUM_KEYPOINTS, 2):
        keypoints = np.hstack((keypoints, np.zeros((NUM_KEYPOINTS, 1))))

    for index, keypoint in enumerate(keypoints):
        x, y, z = keypoint
        string += str(x) + ',' + str(y) + ',' + str(z) + ';'

    return string

# converts a string to keypoints
def from_string(string):
    keypoints = np.zeros((NUM_KEYPOINTS, 3))
    array = string.strip(';').split(';')

    for index, value in enumerate(array):
        x, y, z = [float(num) for num in value.strip(',').split(',')]
        keypoints[index] = (x, y, z)

    return keypoints

# encodes keypoints as data
def encode(keypoints):
    return to_string(keypoints).encode()

# decodes keypoints from data
def decode(data):
    return from_string(data.decode())

# parses file made up of lines containing 21 (x,y) keypoints and a label
def parse(f, shuffle=False, normalization=None, split=None):
    raw = np.array(pd.read_csv(f, sep=',', header=None).values[1:])
    data = raw[:, :-1].astype('float32') 
    labels = raw[:, -1].astype('int8')

    if shuffle is True:
        data, labels = dataset.shuffle(data, labels)        

    for index, entry in enumerate(data):
        if normalization == 'cartesian':
            data[index] = normalize_cartesian(entry)
        elif normalization == 'polar':
            data[index] = normalize_polar(entry)

    if normalization is not None:
        data = data[:, 2:]

    if split is not None:
        split = int(data.shape[0] * split)
    else:
        split = data.shape[0]

    train = dataset(data[:split], labels[:split])
    test = dataset(data[split:], labels[split:])

    return train, test

# displays keypoints
def display(keypoints):
    for index, keypoint in enumerate(keypoints):
        print("keypoint[" + str(index) + "] = " + str(keypoint))
