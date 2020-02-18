import numpy as np
from hand_segmentation import HandSegmetation
import time, cv2, argparse, sys, csv, math
sys.path.append('../cameramouse/')
from cameramouse import RealSenseCamera, WebcamCamera

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", type=str, required=True,
	help="Set to realsense or webcam")
ap.add_argument("-fname", "--filename", type=str, required=True,
	help="Set to wherever you want to save data")
ap.add_argument("-log", "--log", type=bool, required=True,
	help="Set to wherever you want to save data")
ap.add_argument("-src", "--src", type=int, default=0,
	help="src number of camera input")
args = ap.parse_args()

if args.camera == "realsense":
    camera = RealSenseCamera()
elif args.camera == "webcam":
    camera = WebcamCamera(args.src)
else:
    raise NameError("Invalid camera type, must me realsense or webcam")

handSeg = HandSegmetation(camera, testMorphology=False, numRectangles=9, blurKernel=(7,7))
states = np.zeros((300, 10))
start = time.time()
if args.log:
	stop_time = 10
	i = 0
else:
	stop_time = math.inf

if __name__ == "__main__":
    cv2.namedWindow("ColorFeed")
    while (time.time() - start) < stop_time:
        gray_img, color_img = camera.capture_frames()
        handSeg.get_velocity(color_img)
        rect = handSeg.new_state.rectangle
        centroid = (handSeg.new_state.centroid[0], handSeg.new_state.centroid[1])
        # for rect, centroid in rects:
        cv2.rectangle(color_img, (int(rect[0]), int(rect[1])), \
              (int(rect[0]+rect[2]), int(rect[1]+rect[3])), \
               [0, 0, 255], 2)
        cv2.circle(color_img, centroid, 5, [255, 0, 255], -1)

        if (args.log) and (i < 300):
            states[i, 0:4] = handSeg.new_state.rectangle
            states[i, 4:6] = handSeg.new_state.centroid
            states[i, 6:8] = handSeg.new_state.velocity
            states[i, 8] = handSeg.new_state.area
            states[i, 9] = handSeg.new_state.timestamp
            i+=1

        cv2.imshow("ColorFeed", color_img)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            cv2.destroyWindow("ColorFeed")
            break

    if args.log:
        np.savetxt(args.filename, states, delimiter=',')
