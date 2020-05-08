import numpy as np
import time, argparse, cv2, sys, copy, yaml
import hand_tracking.colour_segmentation as colour_segmentation

class HandTracker():
    def __init__(self, camera):
        self.handSeg = colour_segmentation.HandSegmetation(camera, testMorphology=True)
        self.hand = colour_segmentation.Hand() # average 5 most recent positions
        self.prev_hand = colour_segmentation.Hand()

        # constants used during global recognition
        self.dist_threshold = 7500 # how far contour defects need to be from convex hull
        self.area_threshold = 45000 # how big detected hand area needs to be
        
        self.found = 0 # if we know where the hand is
        self.expansion_const = 50 # how much to expand ROI in prediction step

        # Initialise the size of the frame
        gray, color = camera.capture_frames()
        self.y_bound, self.x_bound, _ = color.shape
        self.centre = (self.x_bound // 2, self.y_bound // 2)

        # For writing on frames
        self.writer = {"font": cv2.FONT_HERSHEY_SIMPLEX, "origin": (50, 50), "font_size": 1, "colour": (255, 0, 0), "thickness": 2}

        print("[DEBUG] Hand Tracker Initialised")

    def global_recognition(self, color_frame):
        """
        Description: Processes full frames to initialise the position of the hand, works by detecting an outstretched hand.
        Specifically it looks for defects in the convex hull of the contour which are far enough from the hull. These
        points correspond to the webbing in between each finger.
        Inputs:
            color_frame: returned from camera object, it is a numpy array
        Outputs:
            centroid: centroid of the located hand
        """
        image = cv2.putText(color_frame, 'Outstretch hand to be detected', \
                            self.writer["origin"], self.writer["font"], self.writer["font_size"], \
                            self.writer["colour"], self.writer["thickness"], cv2.LINE_AA)
        
        # locate contours of skin-coloured regions                    
        conts = self.handSeg.get_objects(color_frame)

        # Iterate over all contours and check if they are a hand
        for i, cont in enumerate(conts):
            contArea = cv2.contourArea(cont)
            hull = cv2.convexHull(cont, returnPoints = False)
            defects = cv2.convexityDefects(cont, hull)
            out_count = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                far = tuple(cont[f][0]) #farthest point on contour from the line
                if d > self.dist_threshold:
                    out_count += 1
                
                cv2.circle(color_frame,far,5,[0,0,255],-1)

            # found the hand: initialise the tracker
            if (out_count >= 4) and (contArea > self.area_threshold): # thresholds for hand being found
                # Get bounding region
                rect = np.array(cv2.boundingRect(cont))
                # Get centroid
                moment = cv2.moments(cont)
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                centroid = np.array([cx, cy])
                # Set our initial state
                self.prev_hand.set_state(rect, centroid, contArea, time.time())
                self.found = 1
                return centroid


    def update_position(self, frame, control_type):
        """
        Steps:
            1. Predict where the hand will be using prior velocity
            2. Search the predicted region for the hand and find the centroid
            3. Take a new skin colour sample from around the centroid and adapt the histogram
            4. Draw bounding rectangles on the frame
        Input:
            frame: frame to search
        Outputs:
            centroid: the position of the new hand in the image frame
        """
        # predict hand location based on velocity of hand in prev frame
        box = self._predict_position()

        # only look where we think the hand is
        roi = frame[box[1]:box[3], box[0]:box[2], :]

        # get all contours in roi
        conts = self.handSeg.get_objects(roi)

        # Get all contour areas and store in a numpy array
        areas = np.array([cv2.contourArea(cont) for cont in conts])

        if len(conts) == 0: # no hand - reset
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
            local_centroid = centroid.copy()

            # need to do a coordinate shift as only searching roi not the whole frame
            centroid[0] += box[0]
            centroid[1] += box[1]

            # set the state of the new hand
            self.hand.set_state(rect, centroid, rect_area, time.time())

            # adapt the histogram representation with a new sample from around the hand centroid
            sample_size = 50
            top_left = centroid -  sample_size // 2

            hand_sample = frame[top_left[1]:top_left[1]+sample_size, top_left[0]:top_left[0]+sample_size, :]
            self.handSeg.adapt_histogram(hand_sample)

            # draw ROI bounding box and bounding box for the hand
            cv2.rectangle(frame, (int(box[0]), int(box[1])), \
                  (int(box[2]), int(box[3])), \
                   [0, 0, 255], 2)

            # draw the located hand area
            cv2.rectangle(frame, (int(rect[0]), int(rect[1])), \
                  (int(rect[0]+rect[2]), int(rect[1]+rect[3])), \
                   [0, 255, 0], 2)

            # draw the region we a re resampling from
            # cv2.rectangle(frame, (int(top_left[0]), int(top_left[1])), \
            #       (int(top_left[0]+sample_size), int(top_left[1]+sample_size)), \
            #        [255, 0, 0], 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 5, (255, 0, 0), 1)
            
            if control_type == "joystick":
                cv2.circle(frame, self.centre, 85, (0, 0, 0), 1)
            
            vel = self.hand.centroid - self.prev_hand.centroid
            self.prev_hand = copy.copy(self.hand)
            self.prev_hand.velocity = vel

            return self.hand.centroid

    def _predict_position(self):
        """
        Steps:
            1. Use the old hand position as an initial guess
            2. Translate the box by the prior velocity of the hand
            3. Expand the box by a constant to ensure the whole hand is found
            4. Clip to ensure the box doesn't leave the boundaries
        Outputs:
            corners: the corner locations for the box [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        """
        # set corners to be prev bounding box
        corners = np.zeros((4), dtype=int)
        corners = self.prev_hand.rectangle

        # transform to corner locations
        corners[2] = corners[0] + corners[2]
        corners[3] = corners[1] + corners[3]

        # translate the box by the hands relative movement
        # print("Current: {}\nPrevious: {}".format(self.hand.centroid, self.prev_hand.centroid))

        if self.hand.centroid is None:
            # on startup the new hand has no velocity
            velocity = np.array([0, 0])
        else:
            velocity =  self.prev_hand.velocity

        corners[0:2] = velocity + corners[0:2] 
        corners[2:] = velocity + corners[2:]

        # expand the box size a constant amount
        corners[0:2] -= self.expansion_const
        corners[2:] += self.expansion_const

        corners[0] = np.clip(corners[0], 0, self.x_bound)
        corners[1] = np.clip(corners[1], 0, self.y_bound)
        corners[2] = np.clip(corners[2], 0, self.x_bound)
        corners[3] = np.clip(corners[3], 0, self.y_bound)

        return corners

