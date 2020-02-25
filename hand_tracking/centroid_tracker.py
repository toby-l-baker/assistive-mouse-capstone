import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# Resources
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/


class CentroidTracker():
    def __init__(self, hand_hist, maxDisappeared=20):
        """Insert a while here to get the hand histogram(s)"""
        self.hand_hist = hand_hist
        # initialize ordered dictionaries for objects we are tracking and those that
        # have dissapeared/been occluded.
        self.objects = OrderedDict()
        self.dissapeared = OrderedDict()
        self.nextObjectID = 0

        # how many frames an object can dissapear for before being axed.
        self.maxDisappeared = maxDisappeared

    def registerObject(self, centroid):
        self.objects[self.nextObjectID] = centroid # register object with unique ID
        self.dissapeared[self.nextObjectID] = 0 # object is in current frame so set to 0
        self.nextObjectID += 1 # increment next ID

    def deregisterObject(self, objectID):
        del self.objects[self.objectID]
        del self.dissapeared[self.objectID]

    """
    Updates tracking dictionaries based off current frame info

    Inputs: list of bounding boxes for object in current frame
    Returns: set of trackable objects
    """
    def updateObjects(self, objects): # rects is output from getObjects
        # Check for empty list first (i.e. no rectangles)
        if len(rects) == 0:
            for objectID in list(self.dissapeared.keys()):
                self.dissapeared[objectID] += 1

                if self.dissapeared[objectID] > self.maxDisappeared:
                    self.deregisterObject(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i in range(len(objects)):
            inputCentroids[i] = objects[i].centroid

        # we are currently not tracking anything
        if len(self.objects) == 0:
            for i in range(0, len(objects)):
                self.registerObject(inputCentroids[i])

        # we are currently tracking things and need to do some matching
        else:
            # get the IDs and centroids of the things we are tracking.
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance from each objectCentroid to one of the input
            # centroids and try and make a match
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # find the smalles value in each row and sort indices based on min values
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]


            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                #if we have already checked the column or row, ignore it
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.dissapeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
        return self.objects
