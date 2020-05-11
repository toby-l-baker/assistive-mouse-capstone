# OpenCV Only Full Python  Implementation

### Files of interest to run 

Make sure to edit config.yaml to match your setup and the different control/segmentation type you want to use. A video demo can be seen [here](https://youtu.be/ekWOpIs6XiM). You place your hand in the green square, press 'z' to calibrate the user's skin colour and finally outstretch your hand to have it be detected. After this you can move your hand around to control the cursor. When using the keyboard mouse press 'o' to go out of range and any other key to exit. 'd' to start dragging and 'd' to stop. 's' is single click and the others can be found in gesture_recognition/keyboard.py.

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

1. Hand Tracking: Responsible for keeping track of the hands state (position and velocity information). In particular I have been using hand tracking to predict where the hand will be so I only need to check a small subset of the frame.
2. Hand Segmentation: Just needs to take in an image and return the coordinates of the hand centroid in the frame provided.
3. Filter: Takes in the hands position and is capable of getting the filtered position. Has two functions: insert_point and get_filtered_position.
4. Control: Takes in the filtered positions of the hand and maps this to cursor motion. Also takes in gesture info(type Gestures in gesture_recognition/gestures.py) and maps that to cursor actions such as a click or right-click. 
5. Gesture Recognition: Processes whatever input be it video or voice and outputs the gesture type at any point to the controller module.

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
