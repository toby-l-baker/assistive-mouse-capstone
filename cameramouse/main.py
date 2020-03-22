from cameramouse import OpticalFlowMouse, HandSegmentationMouse
from camera import RealSenseCamera, WebcamCamera
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", type=str, required=True,
	help="Set to realsense or webcam")
ap.add_argument("-m", "--mouse", type=str, required=True,
	help="Options: optical, segmentation")
ap.add_argument("-filt", "--filter", type=str, required=True,
	help="Options: FIR, IIR")
ap.add_argument("-filt_size", "--filter_size", type=int, required=True,
	help="Options: some integer")
ap.add_argument("-src", "--source", type=int, default=0,
	help="src number of camera input")
ap.add_argument("-control", "--control", type=str, default="vel",
	help="type of cursor control: abs, vel, hybrid")
args = ap.parse_args()

if args.camera == "realsense":
    camera = RealSenseCamera()
elif args.camera == "webcam":
    print("Camera Feed {}".format(args.source))
    camera = WebcamCamera(args.source)
else:
    raise NameError("Invalid camera type, must me realsense or webcam")

if args.mouse == "optical":
    mouse = OpticalFlowMouse(camera)
elif args.mouse == "segmentation":
    assert(args.filter.upper() in ["IIR", "FIR"])
    mouse = HandSegmentationMouse(camera, args.filter, args.filter_size, args.control)
else:
    raise NameError("Invalid mouse type")

if __name__ == "__main__":
    mouse.run()
