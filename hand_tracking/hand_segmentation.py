import cv2
import numpy as np
import time

class Hand():
    def __init__(self):
        self.centroid = None
        self.rectangle = None
        self.area = None
        self.velocity = None
        self.timestamp = None

    def set_state(self, rect, centroid, area, velocity, timestamp):
        self.centroid = centroid
        self.rectangle = rect
        self.area = area
        self.velocity = velocity
        self.timestamp = timestamp

    def set_prev_state(self, hand):
        self.centroid = hand.centroid
        self.rectangle = hand.rectangle
        self.area = hand.area
        self.velocity = hand.velocity
        self.timestamp = hand.timestamp

    def update_velocity(self, old_state):
        dt = (self.timestamp - old_state.timestamp)
        dx = -(self.centroid[0] - old_state.centroid[0])
        dy = -(self.centroid[1] - old_state.centroid[1])
        # print("dt {}, dx {}, dy {}".format(dt, dx, dy))
        if dt == 0:
            self.velocity = np.array([dx/(1/30), dy/(1/30)]) # assuming FPS of 30
        else:
            self.velocity = np.array([dx/dt, dy/dt])

class User():
    def __init__(self, username):
        # load in parameters specific to the user
        self.hand_hist = None
        self.open_size = None
        self.closed_size = None

class HandSegmetation():
    def __init__(self, camera, testMorphology=False, numRectangles=9, blurKernel=(7,7)):
        self.camera = camera # to allow this function to get frames
        hand_hist_created = False # flag to see if we have a hand histogram
        self.blurKernel = blurKernel # size of kernel for Gaussian Blurring
        self.numRectangles = numRectangles

        gray_frame, self.color_frame = self.camera.capture_frames()
        self.createRectangles(rect_size=50)

        self.morphElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 16))
        self.denoiseElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.dilationIterations = 4
        self.erosionIterations = 2
        self.colorThresh = 50
        self.areaThreshold = 0

        self.testMorphology = testMorphology
        if self.testMorphology:
            cv2.namedWindow("MorphologyTest")
            cv2.createTrackbar("dilate_iterations", "MorphologyTest", \
              self.dilationIterations, 10, self.updateIterationsCallback)
            cv2.createTrackbar("erosion_iterations", "MorphologyTest", \
              self.erosionIterations, 10, self.updateIterationsCallback) #value, count, function to call
            cv2.createTrackbar("threshold_value", "MorphologyTest", \
              self.colorThresh, 100, self.updateIterationsCallback) #value, count, function to call

        # loop until we have a hand histogram (indicated by pressing 'z')
        cv2.namedWindow("CalibrationFeed")
        while hand_hist_created == False:
            pressed_key = cv2.waitKey(1)
            gray_frame, self.color_frame = self.camera.capture_frames()
            self.blur = cv2.GaussianBlur(self.color_frame, self.blurKernel, 0)
            self.draw_rect() # draw boxes where we will sample hand colour

            if pressed_key & 0xFF == ord('z'):
                hand_hist_created = True
                self.hand_hist = self.hand_histogram()

            cv2.imshow("CalibrationFeed", self.blur)

        # Get all hand coloured objects in first frame
        self.getObjects(self.color_frame)
        # Get max rectangle and st this to be the hand
        max = self.getMaxRectangle()
        self.old_state = Hand() # rect, centroid, area, vel, time
        self.old_state.set_state(max[0], max[1], max[2], 0, time.time())
        self.new_state = Hand()
        cv2.destroyWindow("CalibrationFeed")

    def createRectangles(self, rect_size=100):
        rows, cols, _ = self.color_frame.shape
        self.rect_size = rect_size
        self.hand_rect_one_x = int(9 * rows / 20)
        self.hand_rect_one_y = int(9 * cols / 20)
        # top right corner of the squares
        self.hand_rect_two_x = self.hand_rect_one_x + self.rect_size
        self.hand_rect_two_y = self.hand_rect_one_y + self.rect_size

    def draw_rect(self):
        # for i in range(self.numRectangles):
        cv2.rectangle(self.blur, (self.hand_rect_one_y, self.hand_rect_one_x), (self.hand_rect_two_y, self.hand_rect_two_x), (0, 255, 0), 1)

    def hand_histogram(self):
        hsv_frame = cv2.cvtColor(self.blur, cv2.COLOR_BGR2HSV)
        roi = np.zeros([self.rect_size, self.rect_size, 3], dtype=hsv_frame.dtype) #region of interest

        # load our pixel values from each rectangle into roi variable
        # for i in range(self.numRectangles):
        roi = hsv_frame[self.hand_rect_one_x:self.hand_rect_one_x + self.rect_size, self.hand_rect_one_y:self.hand_rect_one_y + self.rect_size]
        # print("Minimum Hue: {}, Maximum Hue: {}".format(np.min(roi[:,:,0]), np.max(roi[:,:,0])))
        # print("Minimum Sat: {}, Maximum Sat: {}".format(np.min(roi[:,:,1]), np.max(roi[:,:,1])))
        hand_hist = cv2.calcHist([roi], [0, 1], None, [12, 15], [0, 180, 0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def updateIterationsCallback(self, _):
        self.dilationIterations = cv2.getTrackbarPos("dilate_iterations", "MorphologyTest")
        self.erosionIterations = cv2.getTrackbarPos("erosion_iterations", "MorphologyTest")
        self.colorThresh = cv2.getTrackbarPos("threshold_value", "MorphologyTest")

    """
    Takes in the current frame (from main) and returns a list of bounding boxes
    containing all skin coloured objects in the image.
    """
    def getObjects(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cv2.filter2D(dst, -1, disc, dst)
        _, thresh = cv2.threshold(dst,self.colorThresh,255,cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.denoiseElement)
        dilation = cv2.dilate(opening, self.morphElement, iterations=self.dilationIterations)
        # threshObjectSegment = cv2.merge((threshObjectSegment,threshObjectSegment,threshObjectSegment))

        # locatedObject = cv2.bitwise_and(frame, threshObjectSegment)

        # locatedObjectGray = cv2.cvtColor(locatedObject, cv2.COLOR_BGR2GRAY)

        # _, locatedObjectThresh = cv2.threshold(locatedObjectGray, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        locatedObject = cv2.medianBlur(dilation, 5)

        if self.testMorphology:
            cv2.imshow("MorphologyTest", dilation)

        # get contours from thresholded image
        _, cont, hierarchy = cv2.findContours(locatedObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.rectangles = []
        """ TODO: Add geometric filtering to rectangles (ratios etc)"""
        for i, contour in enumerate(cont):
            try:
                rect = np.array(cv2.boundingRect(contour))
                rectArea = cv2.contourArea(contour)
                moment = cv2.moments(contour)
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                centroid = np.array([cx, cy])
                # if rectArea > 75000:
                self.rectangles.append((rect, centroid, rectArea))
                # print("{}: {}".format(i, cv2.contourArea(contour)))
            except Exception as e:
                print(e)

    def getOnlyObjects(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cv2.filter2D(dst, -1, disc, dst)
        _, thresh = cv2.threshold(dst,self.colorThresh,255,cv2.THRESH_BINARY)
        erosion = cv2.erode(thresh, self.denoiseElement, iterations = self.erosionIterations)
        dilation = cv2.dilate(erosion, self.morphElement, iterations=self.dilationIterations)

        locatedObject = cv2.medianBlur(dilation, 5)

        if self.testMorphology:
            cv2.imshow("MorphologyTest", dilation)

        # get contours from thresholded image
        _, cont, hierarchy = cv2.findContours(locatedObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return cont

    def getMaxRectangle(self):
        try:
            max = self.rectangles[0]

            for rect in self.rectangles:
                if rect[2] > max[2]:
                    max = rect
        except Exception as e:
            print(e)
            max = None

        return max

    # """
    # Gets velocity of hand!!!!
    # """
    # def get_velocity(self, frame):
    #     # Finds all the rectangles in our image
    #     self.getObjects(frame)
    #     # Find the largest one
    #     max = self.getMaxRectangle()
    #
    #     if max is not None:
    #         # Update the state of our hand
    #         timestamp = time.time()
    #         self.new_state.set_state(max[0], max[1], max[2], 0, timestamp) # rect, centroid, area, vel, time
    #         self.new_state.update_velocity(self.old_state)
    #         # Get velocities
    #         self.vel_x, self.vel_y = self.new_state.velocity
    #
    #         self.old_state.set_state(max[0], max[1], max[2], self.new_state.velocity, timestamp)
    #
    #     """TODO: Use changes in hand area to ignore movements"""
