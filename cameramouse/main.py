from cameramouse import OpticalFlowMouse, HandSegmentationMouse
from camera import RealSenseCamera, WebcamCamera
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", type=str, required=True,
	help="Set to realsense or webcam")
ap.add_argument("-m", "--mouse", type=str, required=True,
	help="Options: optical, segmentation")
ap.add_argument("-src", "--src", type=int, default=0,
	help="src number of camera input")
args = ap.parse_args()

if args.camera == "realsense":
    camera = RealSenseCamera()
elif args.camera == "webcam":
    camera = WebcamCamera(args.src)
else:
    raise NameError("Invalid camera type, must me realsense or webcam")

if args.mouse == "optical":
    mouse = OpticalFlowMouse(camera)
elif args.mouse == "segmentation":
    mouse = HandSegmentationMouse(camera)
else:
    raise NameError("Invalid mouse type")

if __name__ == "__main__":
    mouse.run()
