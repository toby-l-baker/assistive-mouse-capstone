# assistive-mouse-capstone
Empowering those who cannot use computer mouses/stylus' to be able to use them with only a webcam and some basic gestures.


## MediaPipe
### Installation
Follow the instructions in `mediapipe/README.md`  
Note that this is using MediaPipe v0.7.0 with Bazel v1.2.1

### Building Source
```
cd mediapipe
./scripts/build.sh
```

### Running Demo 1
```
cd mediapipe
./scripts/run_demo1.sh
```
Demo1 uses a frame reduced MediaPipe(described in the capstone report, implemented in commit d98bf446c65d2c046703090f4216b1a2c5ee4469 in mediapipe, note assistive-mouse-capstone/mediapipe itself is also a repo). [gesture_detection_calculator.cc](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mediapipe/mediapipe/calculators/util/gesture_detection_calculator.cc) gets the coordinates of 21 keypoints and uses hand geometry to determine the current gesture, and uses a number to represent it. There are two versions of this file, the old version simply counts the number of scretch fingers, the [new version](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mediapipe/mediapipe/calculators/util/gesture_detection_new_calculator.cc) uses a gesture definition like [this](https://docs.google.com/presentation/d/1R5K-rlorkxrP03RoG5ys7vCLMY0H_5_y4Tqb3lC5Uv8/edit?usp=sharing). Currently if you build MediaPipe you get the old version, you need to mannually substitute the file to get the new version. 

The number representation of the gesture then goes to [pipe_writing_calculator.cc](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mediapipe/mediapipe/calculators/util/pipe_writing_calculator.cc) which calculates the centroid of the 5 picked keypoint (described in the report) and writes it to a fifo together with the gesture.

The [mouse control script](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mouse-control-test/mouse_control_for_demo1_with_new_gesures.py) works with the new version.
### Running Demo 2
```
cd mediapipe
./scripts/run_demo2.sh
```

Demo2 aims to integrate HSV OpenCV Colour Segmentation with MediaPipe. This means calibration can happen automatically when MediaPipe detects the hand. It doens't work super reliably as MediaPipe has false positives. It needs improvements where when mediapipe is laggy, frames still get passed to the OpenCV Segmentation node. I believe it is possible to have some nodes receive data more often than others. It aims to get the benefits of MediaPipe with regard to keypoints for gestures and more speed by using the hand segmentation in parallel with MedaPipe.

### Networking
Protocol: UDP  
IP Address: `127.0.0.1`  
Ports: `2000`, `3000`, `4000`

The 21 key points of the hand can be received over ports `2000` or `3000`.
The data is transmitted as a string formatted as `"x_i, y_i, z_i;"` where `1 <= i <= 21`.
Note that each entry in the tuple is a floating-point number between 0 and 1.

Hand tracking information can be received over port `4000`.
The data is transmitted as a string formatted as `"x_cent, y_cent, x_rect, y_rect, w_rect, h_rect;"`.
`(x_cent, y_cent)` is the location of the centroid of the hand expressed as floating-point numbers.
`(x_rect, y_rect, w_rect, h_rect)` is the location and size of the bounding box expressed as floating-point numbers.
Note that this information is only available when running Demo 2.

## OpenCV

Head to the ```cameramouse/README.md``` to understand more about the purely OpenCV implementation and possible future steps on this particular avenue. Created since this will be portable to all OS's and is lightweight in terms of computation, currently runs at the full 30 FPS. MediaPipe is very promising and provides a lot of good information regarding gestures but is computationally expensive and only works on Linux. Differentiating hand gestures is hard with OpenCV but inputs such as voice could be useful. I believe Weihao had some luck with some voice recognition software called snowboy.

## Gesture Learning
TODO

## Mouse Control
TODO

For mouse control for demo 1, see the comments in the [script](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mouse-control-test/mouse_control_for_demo1_with_new_gesures.py) for details.
## Testing
See the capstone report for details.

## Important Files/Directories
TODO  
`mediapipe/mediapipe/calculators/util/landmark_forwarder_calculator.cc`: forwards 21 key points over UDP  
`mediapipe/mediapipe/calculators/util/hand_tracking_calculator.cc`: hand segmentation and tracking with OpenCV  
`gesture_learning/keypoints.py`: utility functions to manipulate key points generated by MediaPipe  
`gesture_learning/GestureRecognition.py`: runs a gesture learning model in real-time alongside MediaPipe  
`gesture_learning/template.py`: template for training a gesture learning model  
`gesture_learning/data/`: contains gesture datasets  
`gesture_learning/models/`: contains gesture learning models  

## Important OpenCV Implementation files
`cameramouse/config.yaml`:contains the parameters required to load each module along with various tunable constants.
`cameramouse/utils.py`: if you make a new module you will need to edit the Loaders class to include it.
`cameramouse/cameramouse.py`: handles the initialisation and running of all modules, passes data between them.
`cameramouse/control/filters.py`: contains super and subclasses for simple IIR and FIR filters. Work to be done on improving these.
`cameramouse/control/controllers.py`: contains various methods of mapping hand movement to cursor movement, e.g. absolute or relative.
`cameramouse/gestures/`: contains Gestures object as well as an implementation that uses keyboard presses. Work to be done here (voice or hand gesture recognition using OpenCV)
`cameramouse/hand_tracking/`: contains hand segmentation objects and hand tracking objects.
