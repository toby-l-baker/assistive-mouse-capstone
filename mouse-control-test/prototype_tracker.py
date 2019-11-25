import cv2
import pyrealsense2 as rs
import sys
import numpy as np
# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# https://nanonets.com/blog/optical-flow/
# https://gitlab.com/minotnepal/opencv/blob/abe2ea59edbb7671bbf014d308e888db2a9bfab6/samples/python2/lk_track.py
https://ieeexplore.ieee.org/abstract/document/990976

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Variable for color to draw optical flow track
color = (0, 255, 0)

'''Configure depth and colur streams'''
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
'''Start streaming'''
pipeline.start(config)
'''Get initial frame and convert to grayscale'''
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
prev_gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
prev = cv2.goodFeaturesToTrack(prev_gray_image, mask = None, **feature_params)
mask = np.zeros_like(color_image)

try:
    while True:
        '''Wait for a coherent pair of frames: depth and color'''
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if or not color_frame:
            continue
        'Convert images to numpy arrays'''
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray_image, gray_image, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            color_image = cv2.circle(color_image, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(color_image, mask)
        # Create  copy of the gray image
        prev_gray_image = gray_image.copy()

        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', output)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

finally:
    if realsense:
        pipeline.stop()
    cv2.destroyAllWindows()
