"""
Description: Functions to manipulate MediaPipe keypoints
Author: Ayusman Saha
"""
import numpy as np

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
    normalized = np.zeros((NUM_KEYPOINTS, 3))
    origin = keypoints[0, :-1]
    fixed = keypoints[9, :-1]
    theta = angle(fixed - origin) + np.pi/2

    for index, keypoint in enumerate(keypoints):
        normalized[index, :-1] = rigid_body_transform(keypoint[:-1], origin, theta) * np.array([1, -1])

    return normalized

# normalizes keypoints in polar coordinates
def normalize_polar(keypoints):
    normalized = np.zeros((NUM_KEYPOINTS, 3))
    origin = keypoints[0, :-1]
    fixed = keypoints[9, :-1]
    theta = angle(fixed - origin) + np.pi/2

    for index, keypoint in enumerate(keypoints):
        vector = keypoint[:-1] - origin
        normalized[index, :-1] = np.array([length(vector), angle(vector) - theta]) * np.array([1, -1])

    return normalized

# encodes keypoints as data
def encode(keypoints):
    data = ""

    for i in range(NUM_KEYPOINTS):
        x, y, z = keypoints[i]
        data += str(x) + ',' + str(y) + ',' + str(z) + ';'

    return data.encode()

# decodes keypoints from data
def decode(data):
    keypoints = np.zeros((NUM_KEYPOINTS, 3))
    array = data.decode().strip(';').split(';')
    
    for index, value in enumerate(array):
        x, y, z = [float(i) for i in value.strip(',').split(',')]
        keypoints[index] = (x, y, z)

    return keypoints

# displays keypoints
def display(keypoints, z=False):
    for index, keypoint in enumerate(keypoints):
        if z is False:
            print("keypoint[" + str(index) + "] = " + str(keypoint[:-1]))
        else:
            print("keypoint[" + str(index) + "] = " + str(keypoint))