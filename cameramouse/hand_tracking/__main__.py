import argparse
from hardware.camera import *
from hand_tracking.tracking import *

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("-src", "--src", type=int, default=0, help="src number of camera input")
    args = ap.parse_args()
    
    if args.webcam and args.realsense:
        raise ValueError('Only select one camera type')
    elif not (args.webcam or args.realsense):
        raise ValueError('Select a camera type using --webcam or --realsense')

    return args

args = parse_args()
camera = None
if args.webcam:
    camera = WebcamCamera(args.src)
elif args.realsense:
    camera = RealSenseCamera()

cv2.namedWindow("Tracker Feed")
hand_tracker = HandTracker(camera)
recalibrated = False

while True:
    # grab frames
    gray_frame, color_frame = camera.capture_frames()

    if not hand_tracker.found: # hand is lost
        if not recalibrated:
            hand_tracker.handSeg.get_histogram()
            recalibrated = True
        else:
            hand_tracker.global_recognition(color_frame)
    else: # found the hand lets track it
        hand_tracker.update_position(color_frame, " ")
        recalibrated = False
    cv2.imshow("Tracker Feed", color_frame)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        cv2.destroyWindow("Tracker Feed")
        break