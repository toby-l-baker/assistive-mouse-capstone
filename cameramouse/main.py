from cameramouse import HandSegmentationMouse
from camera import RealSenseCamera, WebcamCamera
from interface import WindowsMouse, WindowsMonitor, LinuxMonitor, LinuxMouse, Mouse
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", type=str, required=True,
	help="Set to realsense or webcam")
ap.add_argument("-filt", "--filter", type=str, required=True,
	help="Options: FIR, IIR")
ap.add_argument("-filt_size", "--filter_size", type=int, required=True,
	help="Options: some integer")
ap.add_argument("-src", "--source", type=int, default=0,
	help="src number of camera input")
ap.add_argument("-os", "--os", type=str, default="linux",
	help="your operating system linux or windows")
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

if args.os == "linux":
    mouse = LinuxMouse()
    monitor = LinuxMonitor()
elif args.os == "windows":
    mouse = WindowsMouse()
    monitor = WindowsMonitor()
else:
    raise NameError("Unsupported OS, must me linux or windows")

assert(args.filter.upper() in ["IIR", "FIR"])
mouse = HandSegmentationMouse(camera, args.filter, args.filter_size, args.control, mouse, monitor)

if __name__ == "__main__":
    mouse.run()
