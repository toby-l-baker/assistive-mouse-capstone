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
### Dataset
gesture_learning/data stores all dataset we created. Each line in dataset consists (x, y) coordinates of 21 keypoints and its class number. We denote its class number in braces. twoClass includes close (0) and open (1) gestures; threeClass includes close (0), OK (1), open(2) gestures; fourClass includes close (0), open (1), OK (2) and click (index finger bent, 3) gestures; fiveClass includes close (0), open (1), scroll_down (index finger stretching, 2), scroll_up (ndex and middle fingers stretching, 3) and slow (thumb stretching, 4). See vedio example for more concrete gesture examples.

### Models
gesture_learning/models stores all models we trained for five models. There are KMeans, Gaussian Mixture, Random Forest and a simple DNN we tried. To be noticed, KMeans, Gaussian Mixture, Random Forest models are kept in joblib formate (use joblib_model= joblib.load(file_path) to load). DNN model is kept in h5 format.

### Train
gesture_learning/learn.py reads in raw dataset mentioned in the previous part, generates augmented features and runs KMeans/Gaussian Mixture/Random Forest Model. The augmented features are strecthing finger numbers, angles between finger pairs and finger distance ratio. See comments in code for more information. We also provide functionality that get mean validation accuracy in several epochs and save models for other use (intergrated in Mediapipe, i.e. joblib.dump(modle, file_path)). Moreover, it is easy to add parser and args to make the learn.py more flexible in terms of file_path and choice of model.

## Mouse Control
TODO

For mouse control for demo 1, see the comments in the [script](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/mouse-control-test/mouse_control_for_demo1_with_new_gesures.py) for details.
## Testing
See the capstone report for details.

## Depth Camera
We have an Intel Realsense depth camera that is able to get extra depth information. The camera should work with [IntelRealsense](https://github.com/IntelRealSense/librealsense). The repository contains many depth camera examples such as outputting depth information and point cloud. It also contains a OpenCV implementation of DNN. To compile the examples, one can follow the instructions in readme from that repo.

We did some reseach of the depth camera, some infomation can be found in this [slides](https://docs.google.com/presentation/d/1SyncibUJNlsJfWg0QvKgYm_z7swszJZDxj19tprEECY/edit?usp=sharing) as well as the report.

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
`cameramouse/control/filters.py`: contains super and subclasses for simple IIR and FIR filters.
`cameramouse/control/controllers.py`: contains various methods of mapping hand movement to cursor movement.
`cameramouse/gestures/`: contains Gestures object as well as an implementation that uses keyboard presses.
`cameramouse/hand_tracking/`: contains hand segmentation objects and hand tracking objects.
