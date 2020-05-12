# OpenCV Only Full Python  Implementation

### Dependencies
```
pip install keyboard==0.13.4 \
            numpy==1.16.5 \ 
            matplotlib==3.1.1 \
            opencv-python pyyaml==5.3.1 \
            argparse \
            pyrealsense2==2.29.0.1124 \
            mouse==0.7.1 \
            pyautogui==0.9.48
```

### Files of interest to run 

Make sure to edit ```config.yaml``` to match your setup and the different control/segmentation type you want to use. A video demo can be seen [here](https://youtu.be/ekWOpIs6XiM). You place your hand in the green square, press 'z' to calibrate the user's skin colour and finally outstretch your hand to have it be detected. After this you can move your hand around to control the cursor. When using the keyboard mouse press 'o' to go out of range and any other key to exit this state. 'd' to start dragging and 'd' to stop. 's' is single click and the other actions can be found in gesture_recognition/keyboard.py.

```
cd cameramouse
python3 main.py 
```

To run only the hand tracking module
```
cd cameramouse
python3 -m hand_tracking --webcam -src 0
or
python3 -m hand_tracking --realsense
```

### System Overview

1. Hand Tracking 
    1. Responsible for keeping track of the hands state (position and velocity information). 
    2. In particular I have been using hand tracking to predict where the hand will be so I only need to check a small subset of the frame.
2. Hand Segmentation: 
    1. Just needs to take in an image and return the coordinates of the hand centroid in the frame provided.
    2. Colour segmentation is sensitive to the calibration & lighting. If your encountering issues just pull your hand away and recalibrate.
    3. Another issue with colour segmentation is that it performs poorly when the hand is moved to the edges of the frame. On calibration it may be worth taking several hand samples, some from the corners and others from the centre of the frame.
    3. The tracker calls ```adapt_histogram``` and passes a square subsection of the frame that surrounds the centroid. This attempts to account for changes in lighting but mostly makes it perform worse. Perhaps because the region is too small. The histogram representation of the skin is updated by a rate according to ```self.alpha```.
    4. Other hand location techniques should definitely be tested such as classifiers or anything else that seems feasible.
3. Filter 
    1. Takes in the hands position and is capable of getting the filtered position. 
    2. Has two primary functions: ```insert_point``` and ```get_filtered_position```.
4. Control
    1. Takes in the filtered positions of the hand and maps this to cursor motion. 
    2. Also takes in gesture info(type Gestures in gesture_recognition/gestures.py) and maps that to cursor actions such as a click or right-click. 
5. Gesture Recognition
    1. Processes whatever input be it video or voice and outputs the gesture type at any point to the controller module.

![cameramouse-system-diagram](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/cameramouse/cameramouse-system-diagram.PNG)

### File Structure

1. hardware/
    1. camera.py: captures image frames
    2. monitor: captures width and height of system info 
    3. mouse: api interface's for different OS's
2. hand_tracking/
    1. colour_segmentation.py: performs hsv colour segmentation on frames 
    2. tracking.py: tracks hand movement and predicts where it will be and passes predictions to segmentation module.
3. gesture_recognition/
    1. gestures.py: containes Gestures class and GestureRecognition parent class
    2. keyboard.py: using key presses to map to gestures
4. control/
    1. controllers.py
    2. filters.py: IIR and FIR filter: very basic implementation with constant scale factors for each value in the stored array - could easily be improved 
    3. mouse_states.py: state machine which handles what gets execute in different states and the transitions: the states are OUT_OF_RANGE - don't execute mouse actions or movements, IN_RANGE - track movement and execute clicks, DRAG - click down on entry and then track movement, on exit move the mouse up.
5. test/
    1. was used to test out different filters on recorded data and simulates cursor movement in matplotlib
5. utils.py
    1. a loading class for loading different modules eg iir vs fir filter
6. cameramouse.py
    1. brings together all of the above 

### Future Work

1. Anti-Shake Filtering for people with tremors
2. Voice recognition for mouse actions - CV tracking works reasonably particularly with relative control and IIR filtering and could easily be adapted to use this over the keyboard input for mouse actions.
3. Other hand segmentation methods
4. OpenCV Gesture Recognition 
