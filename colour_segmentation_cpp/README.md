### Instructions for using this

1. mkdir -p build && cd build
2. cmake .. 
3. make 
4. ./hand_tracker

### Overview

This is simply a c++ implementation of ```../cameramouse/hand_tracking/```. It was created to make the OpenCV code integrable with mediapipe. The resulting mediapipe code can be found in ```../mediapipe/mediapipe/calculators/util/hand_tracking_calculator.cc```. To view the mediapipe graph to understand how this works go (here)[https://viz.mediapipe.dev/] and paste the contents of ```../mediapipe/mediapipe/graphs/hand_tracking/gesture_recognition.pbtxt``` into it.
