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
    def __init__(self, camera, testMorphology=True, numRectangles=9, blurKernel=(7,7)):
        self.camera = camera # to allow this function to get frames
        hand_hist_created = False # flag to see if we have a hand histogram
        self.blurKernel = blurKernel # size of kernel for Gaussian Blurring
        self.numRectangles = numRectangles

        gray_frame, self.color_frame = self.camera.capture_frames()
        self.createRectangles()

        self.morphElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 20))
        self.denoiseElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.dilationIterations = 7
        self.erosionIterations = 1
        self.colorThresh = 5
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

    def createRectangles(self):
        rows, cols, _ = self.color_frame.shape
        #define bottom left corner of the squares
        self.hand_rect_one_x = np.array(
            [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
             12 * rows / 20, 12 * rows / 20], dtype=np.uint32)
        self.hand_rect_one_y = np.array(
            [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
             10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

        # top right corner of the squares
        self.hand_rect_two_x = self.hand_rect_one_x + 10
        self.hand_rect_two_y = self.hand_rect_one_y + 10

    def draw_rect(self):
        for i in range(self.numRectangles):
            cv2.rectangle(self.blur, (self.hand_rect_one_y[i], self.hand_rect_one_x[i]),
                          (self.hand_rect_two_y[i], self.hand_rect_two_x[i]),
                          (0, 255, 0), 1)

    def hand_histogram(self):
        hsv_frame = cv2.cvtColor(self.blur, cv2.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype) #region of interest

        # load our pixel values from each rectangle into roi variable
        for i in range(self.numRectangles):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[self.hand_rect_one_x[i]:self.hand_rect_one_x[i] + 10,
                                              self.hand_rect_one_y[i]:self.hand_rect_one_y[i] + 10]

        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def updateIterationsCallback(self, _):
        self.dilationIterations = cv2.getTrackbarPos("dilate_iterations", "MorphologyTest")
        self.erosionIterations = cv2.getTrackbarPos("erosion_iterations", "MorphologyTest")
        self.colorThresh = cv2.getTrackbarPos("threshold_value", "MorphologyTest")
        self.lower_threshold = cv2.getTrackbarPos("upper_threshold", "CannyEdges")
        self.upper_threshold = cv2.getTrackbarPos("lower_threshold", "CannyEdges")

    """
    Takes in the current frame (from main) and returns a list of bounding boxes
    containing all skin coloured objects in the image.
    """
    def getObjects(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        ret, thresh = cv2.threshold(dst, self.colorThresh, 255, cv2.THRESH_BINARY)

        # Group together adjacent points
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.denoiseElement)
        dilation = cv2.dilate(opening, self.morphElement, iterations=self.dilationIterations)
        frame_threshold = cv2.inRange(hsv, (0, 48, 0), (20, 255, 255))

        if self.testMorphology:
            cv2.imshow("MorphologyTest", frame_threshold)
            cv2.imshow("ThresholdHistogram", dilation)
            # print(np.max(dst))

        # thresh = cv2.merge((thresh, thresh, thresh))
        # [cv2.bitwise_and(frame, thresh), thresh]

        # get contours from thresholded image
        _, cont, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    """
    Gets velocity of hand!!!!
    """
    def get_velocity(self, frame):
        # Finds all the rectangles in our image
        self.getObjects(frame)
        # Find the largest one
        max = self.getMaxRectangle()

        if max is not None:
            # Update the state of our hand
            timestamp = time.time()
            self.new_state.set_state(max[0], max[1], max[2], 0, timestamp) # rect, centroid, area, vel, time
            self.new_state.update_velocity(self.old_state)
            # Get velocities
            self.vel_x, self.vel_y = self.new_state.velocity

            self.old_state.set_state(max[0], max[1], max[2], self.new_state.velocity, timestamp)

        """TODO: Use changes in hand area to ignore movements"""
