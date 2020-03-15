import numpy as np
import time, argparse, cv2, sys
from hand_segmentation import HandSegmetation, Hand, User
sys.path.append('../cameramouse/')
from cameramouse import RealSenseCamera, WebcamCamera

class Filter():
    def __init__(self, size):
        self.positions = np.zeros((size, 2))
        self.filter_size = size

    """Push a new detected centroid onto the stack"""
    def insert_point(self, point):
        self.positions[1:,:] = self.positions[:-1,:]
        self.positions[0, :] = point

    def get_filtered_position(self):
        pass

class FIRFilter(Filter):
    def __init__(self, size):
        super().__init__(size)

    """Returns the averaged position of our x most recent detections"""
    def get_filtered_position(self, _):
        p_x = np.average(self.positions[:, 0])
        p_y = np.average(self.positions[:, 1])
        return np.array([int(p_x), int(p_y)])

class IIRFilter(Filter):
    def __init__(self, size, alpha):
        # assert(len(alpha) == (size))
        super().__init__(size)
        self.alpha = alpha

    def get_filtered_position(self, centroid):
        p_x = (self.alpha/self.filter_size) * np.sum(self.positions[:, 0]) + (1-self.alpha)*centroid[0]
        p_y = (self.alpha/self.filter_size) * np.sum(self.positions[:, 1])  + (1-self.alpha)*centroid[1]
        return np.array([p_x, p_y])

class HandTracker():
    def __init__(self, camera, buf_size, filter, alpha=0.7):
        self.handSeg = HandSegmetation(camera, testMorphology=True)
        self.hand = Hand() # average 5 most recent positions
        self.prev_hand = Hand()
        self.cam = camera
        self.found = 0 # if we know where the hand is
        self.expansion_const = 35 # how much to expand ROI
        self.area_filt_thresh = 1000000

        self.filt_type = filter
        assert(filter in ["IIR", "FIR"])
        if filter == "IIR":
            self.filter = IIRFilter(buf_size, alpha)
        elif filter == "FIR":
            self.filter = FIRFilter(buf_size)

        gray, color = camera.capture_frames()
        self.y_bound, self.x_bound, _ = color.shape

        #for global global_recognition
        self.dist_threshold = 10000
        self.area_threshold = 50000


        # For writing on frames
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (50, 50)
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 2

    """ A function to globally find the hand when it has been lost"""
    def global_recognition(self, color_frame):
        # self.hand.set_state(None, None, None, None, None)
        # while self.hand.centroid == None:
        image = cv2.putText(color_frame, 'Outstretch hand to be detected', \
             self.org, self.font, self.fontScale, self.color, self.thickness, \
             cv2.LINE_AA)
        conts = self.handSeg.getOnlyObjects(color_frame)

        # areas = np.array([cv2.contourArea(cont) for cont in conts])
        # Iterate over all contours and check if they are a hand
        for i, cont in enumerate(conts):
            contArea = cv2.contourArea(cont)
            # print("Contour Area: {}".format(contArea))
            # print("Convexity: {}".format(k))
            hull = cv2.convexHull(cont,returnPoints = False)
            defects = cv2.convexityDefects(cont,hull)
            out_count = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                # start = tuple(cont[s][0]) #start of line
                # end = tuple(cont[e][0]) #end of line
                far = tuple(cont[f][0]) #farthest point on contour from the line
                if d > self.dist_threshold:
                    out_count += 1
                # cv2.line(color_frame,start,end,[0,255,0],2)
                cv2.circle(color_frame,far,5,[0,0,255],-1)
                print("Count: {}".format(out_count))

            if (out_count >= 5) and (contArea > self.area_threshold): # thresholds for hand being found
                # Get bounding region
                rect = np.array(cv2.boundingRect(cont))
                # Get centroid
                moment = cv2.moments(cont)
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                centroid = np.array([cx, cy])
                # Set our initial state
                self.prev_hand.set_state(rect, centroid, contArea, np.array([0, 0]), time.time())
                self.found = 1
                # Load up our array of positions to be used for averaging
                for i in range(len(self.filter.positions)):
                    self.filter.positions[i,0] = centroid[0]
                    self.filter.positions[i,1] = centroid[1]
                break

    """Check some subset of the current frame to locate the hand
    If the hand is lost for more than self.lost_thresh, we will need
    to call global_recognition again"""
    def update_position(self, frame):
        # predict hand location based on velocity of hand in prev frame
        box = self.predict_position()

        # only look where we think the hand is
        roi = frame[box[1]:box[3], box[0]:box[2], :]

        # get all contours in roi
        conts = self.handSeg.getOnlyObjects(roi)

        # Get all contour areas and store in a numpy array
        areas = np.array([cv2.contourArea(cont) for cont in conts])

        if len(conts) == 0: # bye bye
            self.found = 0
        else: # multiple objects - get largest
            maxI = np.where(areas == np.amax(areas))[0]
            maxI = maxI[0]
            rect_area = areas[maxI]
            cont = conts[maxI]

            # get bounding rectangle
            rect = np.array(cv2.boundingRect(cont))

            # need to do a coordinate shift as only searching roi
            rect[0] += box[0] # rect in global coords
            rect[1] += box[1]

            # get centroid
            m = cv2.moments(cont)
            centroid = np.array([int(m['m10']/m['m00']), int(m['m01']/m['m00'])])

            # need to do a coordinate shift as only searching roi
            centroid[0] += box[0]
            centroid[1] += box[1]

            # set state of our new hand
            if self.filt_type == "IIR":
                # push new centroid onto the stack
                self.filter.insert_point(self.prev_hand.centroid)
                self.hand.set_state(rect, self.filter.get_filtered_position(centroid), rect_area, np.array([0, 0]), time.time())
            else:
                # push new centroid onto the stack
                self.filter.insert_point(centroid)
                self.hand.set_state(rect, self.filter.get_filtered_position(centroid), rect_area, np.array([0, 0]), time.time())

            # draw ROI bounding box and boundinng box for the hand
            cv2.rectangle(frame, (int(box[0]), int(box[1])), \
                  (int(box[2]), int(box[3])), \
                   [0, 0, 255], 2)
            rows = int(1/3*(rect[3])) + (rect[1] - box[1]) # 1/3 of height of bounding box + pos of bounding box in ROI
            cols = int(1/3*(rect[2])) + (rect[0] - box[0])# 1/3 of width of bounding box
            hand_sample = roi[rows:rows+50, cols:cols+50, :]
            self.handSeg.adapt_histogram(hand_sample) # pass in lower left corner and frame
            cv2.rectangle(frame, (int(rect[0]), int(rect[1])), \
                  (int(rect[0]+rect[2]), int(rect[1]+rect[3])), \
                   [0, 255, 0], 2)
            cv2.rectangle(frame, (int(cols+box[0]), int(rows+box[1])), \
                  (int(cols+box[0]+50), int(rows+box[1]+50)), \
                   [0, 255, 0], 2)

    """Based on our hands previous position, velocity and box size define a
    bounded region to search for the hand in the next frame. If we are beginning
    to lose track expand the region of interest"""
    def predict_position(self):
        # move corners
        corners = np.zeros((4), dtype=int)
        corners = self.prev_hand.rectangle

        # transform to corner locations
        corners[2] = corners[0] + corners[2]
        corners[3] = corners[1] + corners[3]
        velocity = self.prev_hand.velocity * (1/20) # (self.filter.positions[-1, :] - self.filter.positions[-2, :])
        corners[0:2] = -velocity + corners[0:2] # self.prev_hand.velocity*(1/20)
        corners[2:] = -velocity + corners[2:]

        # expand the box size a constant amount
        corners[0:2] -= self.expansion_const
        corners[2:] += self.expansion_const

        corners[0] = np.clip(corners[0], 0, self.x_bound)
        corners[1] = np.clip(corners[1], 0, self.y_bound)
        corners[2] = np.clip(corners[2], 0, self.x_bound)
        corners[3] = np.clip(corners[3], 0, self.y_bound)

        return corners


    """Get the velocity of our hand based off of previous detections"""
    def get_velocity(self, frame):
        self.update_position(frame)
        self.hand.update_velocity(self.prev_hand)
        # get differences before we set the new state
        dt = self.hand.timestamp - self.prev_hand.timestamp
        darea = abs(self.hand.area - self.prev_hand.area)

        if darea/dt < self.area_filt_thresh:
            self.vel_x = self.hand.velocity[0]
            self.vel_y = self.hand.velocity[1]
        else:
            print("Setting vel to zero")
            self.vel_x = 0
            self.vel_y = 0
        self.prev_hand.set_prev_state(self.hand)

    """Get the velocity of our hand based off of previous detections"""
    def get_position(self, frame):
        self.update_position(frame)
        self.hand.update_velocity(self.prev_hand)
        self.pos_x = self.hand.centroid[0]
        self.pos_y = self.hand.centroid[1]
        self.prev_hand.set_prev_state(self.hand)
        # print("HAND IS AT {} {}".format(self.pos_x, self.pos_y))

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-cam", "--camera", type=str, required=True,
    	help="Set to realsense or webcam")
    ap.add_argument("-src", "--src", type=int, default=0,
    	help="src number of camera input")
    args = ap.parse_args()
    assert((args.camera == "realsense") or (args.camera == "webcam"))

    return args

if __name__ ==  "__main__":
    args = parse_args()
    if args.camera == "realsense":
        camera = RealSenseCamera()
    elif args.camera == "webcam":
        camera = WebcamCamera(args.src)

    cv2.namedWindow("FeedMe")
    handTracker = HandTracker(camera, 5)

    while True:
        # grab frames
        gray_frame, color_frame = camera.capture_frames()

        if not handTracker.found: # hand is lost
            handTracker.global_recognition(color_frame)
        else: # found the hand lets track it
            handTracker.get_velocity(color_frame)
        cv2.imshow("FeedMe", color_frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            cv2.destroyWindow("FeedMe")
            break
