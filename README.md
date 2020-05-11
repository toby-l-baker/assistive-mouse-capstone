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

### Running Demo 2
```
cd mediapipe
./scripts/run_demo2.sh
```

Demo2 aims to integrate HSV OpenCV Colour Segmentation with MediaPipe. This means calibration can happen automatically when MediaPipe detects the hand. It doens't work super reliably as MediaPipe has false positives. It needs improvements where when mediapipe is laggy, frames still get passed to the node. I believe it is possible to have some nodes receive data more often than others. It aims to get the benefits of MediaPipe with regard to keypoints for gestures and more speed by using the hand segmentation.

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

## Testing
TODO

## Important Files/Directories
TODO  
`mediapipe/mediapipe/calculators/util/landmark_forwarder_calculator.cc`: forwards 21 key points over UDP  
`mediapipe/mediapipe/calculators/util/hand_tracking_calculator.cc`: hand segmentation and tracking with OpenCV  
`gesture_learning/keypoints.py`: utility functions to manipulate key points generated by MediaPipe  
`gesture_learning/GestureRecognition.py`: runs a gesture learning model in real-time alongside MediaPipe  
`gesture_learning/template.py`: template for training a gesture learning model  
`gesture_learning/data/`: contains gesture datasets  
`gesture_learning/models/`: contains gesture learning models  
