"""
Description: Functions to manipulate MediaPipe keypoints
Author: Ayusman Saha (aysaha@berkeley.edu)
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

    if normalization == 'cartesian':
        data = np.apply_along_axis(normalize_cartesian, 1, data)
        data = data[:, 2:]
    elif normalization == 'polar':
        data = np.apply_along_axis(normalize_polar, 1, data)
        data = data[:, 2:]
    elif normalization == 'features':
        data = np.apply_along_axis(keypointsToFeatures, 1, data)

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

'''
This function will convert a set of keypoints to a set of features.
param keypoints: numpy array of 21 (x,y) keypoints
return features: numpy array of 20 features
'''
def keypointsToFeatures(keypoints):
    # construct feature matirx
    features = np.zeros(20)

    # distance ratio features
    for i in range(5):
        denominator = (keypoints[8*(i+1) - 1] - keypoints[0])**2 + (keypoints[8*(i+1)] - keypoints[1])**2  # distance from tip to palm
        for j in range(3):
            numerator = (keypoints[8*i + 2*j + 1] - keypoints[0])**2 + (keypoints[8*i + 2*j + 2] - keypoints[1])**2  # distance from root/mid1/mid2 to palm
            ratio = np.sqrt(numerator) / np.sqrt(denominator)
            features[i*3 + j] = ratio
            # features[i*3 + j] = ratio * 10  # 10 times to make more spearable?

    # finger number feature
    for i in range(len(features)):
        features[15] = sum(features[3 : 15 : 4] <= 1) * 10  # stretch finger number, weighted by 10

    # angle features
    for i in range(4):  # four pairs
        x1 = np.array([keypoints[8*(i+1) - 1] - keypoints[0], keypoints[8*(i+1)] - keypoints[1]], dtype=np.float32).T
        x2 = np.array([keypoints[8*(i+2) - 1] - keypoints[0], keypoints[8*(i+2)] - keypoints[1]], dtype=np.float32).T
        cos = np.sum(x1*x2) / (np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)))  # caculate cos(theta)

        # when zero angle, it is possible
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        features[16 + i] = (np.arccos(cos) / np.pi * 180)
        # features[16 + i] = (np.arccos(cos) / np.pi * 180)**2  # Note: use quadratic here

    # return feature matrix
    return features
