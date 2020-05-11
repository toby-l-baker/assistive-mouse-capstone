import cv2, yaml
import numpy as np
import time

class Hand():
    """
    Class to represent the hand with member variables and helper functions to update the state
    """
    def __init__(self):
        self.centroid = None
        self.rectangle = None
        self.area = None
        self.timestamp = None
        self.velocity = np.array([0, 0])

    def set_state(self, rect, centroid, area, timestamp):
        """
        Sets the state of the hand using output of CV functions
        """
        self.centroid = centroid
        self.rectangle = rect
        self.area = area
        self.timestamp = timestamp

class HandSegmetation():
    """
    Hand Segmentation Class for HS Colour Segmentation
    """
    def __init__(self, camera, testMorphology=False, numRectangles=9, blurKernel=(7,7)):
        self.blurKernel = blurKernel # size of kernel for Gaussian Blurring
        self.numRectangles = numRectangles
        self.testMorphology = testMorphology
        self.camera = camera # needed for the segmentation to calibrate
        _, self.color_frame = self.camera.capture_frames()

        # create a rectangle to sample skin colour from
        self.sample_rect_size = 100
        centre = [self.color_frame.shape[1] // 2, self.color_frame.shape[0] // 2 - 100]
        self.sample_rectangle = self.create_rectangle(centre, self.sample_rect_size)

        self.morphElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 16))
        self.denoiseElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.dilationIterations = 4
        self.erosionIterations = 2
        self.colorThresh = 50
        self.areaThreshold = 0
        self.alpha = 0.0
        print("[DEBUG] Colour Segmentation Module Initialized")


    def get_histogram(self):
        """
        Loops until the user presses 'z', after that it'll parse the region in the green square to get an HSV histogram representation 
        of the users skin colour.
        """
        # loop until we have a hand histogram (indicated by pressing 'z')
        cv2.namedWindow("CalibrationFeed")
        hand_hist_created = False
        while hand_hist_created == False:
            pressed_key = cv2.waitKey(1)
            gray_frame, self.color_frame = self.camera.capture_frames()
            self.blur = cv2.GaussianBlur(self.color_frame, self.blurKernel, 0)
            cv2.rectangle(self.color_frame, self.sample_rectangle[0], self.sample_rectangle[1], (0, 255, 0), 1)

            if pressed_key & 0xFF == ord('z'):
                hand_hist_created = True
                self.hand_hist = self.hand_histogram()

            cv2.imshow("CalibrationFeed", self.color_frame)

        cv2.destroyWindow("CalibrationFeed")

    def create_rectangle(self, centre, rect_size):
        """
        Used to create sqaures given a centre point and the size
        Returns a rectangle: [top_left_coords, bottom_right_coords]
        """
        x, y = centre
        tl_x = x - (rect_size // 2)
        tl_y = y - (rect_size // 2)
        # top right corner of the squares
        br_x =  tl_x + rect_size
        br_y = tl_y + rect_size
        return [(tl_x, tl_y), (br_x, br_y)]

    def hand_histogram(self):
        """
        Returns the cv histogram in HSV colour space from the self.sample_rectangle
        """
        hsv_frame = cv2.cvtColor(self.blur, cv2.COLOR_BGR2HSV)
        roi = np.zeros([self.sample_rect_size, self.sample_rect_size, 3], dtype=hsv_frame.dtype) #region of interest

        # load our pixel values from each rectangle into roi variable
        roi = hsv_frame[self.sample_rectangle[0][1]:self.sample_rectangle[1][1], self.sample_rectangle[0][0]:self.sample_rectangle[1][0]]
        hand_hist = cv2.calcHist([roi], [0, 1], None, [12, 15], [0, 180, 0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def adapt_histogram(self, sample):
        """
        Takes in an RGB sample from an image and updates the hand histogram at rate self.alpha
        """
        roi = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        try:
            hand_hist_new = cv2.calcHist([roi], [0, 1], None, [12, 15], [0, 180, 0, 256])
            hand_hist_new = cv2.normalize(hand_hist_new, hand_hist_new, 0, 255, cv2.NORM_MINMAX)
            self.hand_hist = hand_hist_new * self.alpha + (1-self.alpha) * self.hand_hist
            self.hand_hist = cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)
        except:
            pass

    def get_objects(self, frame):
        """
        Takes in a frame and returns the contours of all of the skin coloured regions in that area
        """
        blur = cv2.GaussianBlur(frame, self.blurKernel, 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cv2.filter2D(dst, -1, disc, dst)
        _, thresh = cv2.threshold(dst,self.colorThresh,255,cv2.THRESH_BINARY)
        erosion = cv2.erode(thresh, self.denoiseElement, iterations = self.erosionIterations)
        dilation = cv2.dilate(erosion, self.morphElement, iterations = self.dilationIterations)

        locatedObject = cv2.medianBlur(dilation, 5)

        if self.testMorphology:
            cv2.imshow("MorphologyTest", locatedObject)

        # get contours from thresholded image
        try:
            _, cont, hierarchy = cv2.findContours(locatedObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            cont, hierarchy = cv2.findContours(locatedObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cont
