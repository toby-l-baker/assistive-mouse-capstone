import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# Resources
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/


class CentroidTracker():
    def __init__(self, hand_hist, maxDisappeared=50):
        """Insert a while here to get the hand histogram(s)"""
        self.hand_hist = hand_hist
        # initialize ordered dictionaries for objects we are tracking and those that
        # have dissapeared/been occluded.
        self.ojects = OrderedDict()
        self.dissapeared = OrderedDict()

        # how many frames an object can dissapear for before being axed.
        self.maxDisappeared = maxDisappeared

    def registerObject(self, centroid):
        self.objects[self.nectObjectID] = centroid # register object with unique ID
        self.dissapeared[self.nectObjectID] = 0 # object is in current frame so set to 0
        self.nextObjectID += 1 # increment next ID

    def deregisterObject(self, objectID):
        del self.objects[self.objectID]
        del self.dissapeared[self.objectID]

    """
    Updates tracking dictionaries based off current frame info

    Inputs: list of bounding boxes for object in current frame
    Returns: set of trackable objects
    """
    def updateObjects(self, rects):
        # Check for empty list first
        pass
