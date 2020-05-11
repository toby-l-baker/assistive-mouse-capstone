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

![cameramouse-system-diagram](https://github.com/toby-l-baker/assistive-mouse-capstone/blob/master/cameramouse/cameramouse-system-diagram.PNG)

### File Structure

1. hardware/
2. hand_tracking/
3. gesture_recognition/
4. control/
5. utils.py
6. cameramouse.py

### Future Work

1. Anti-Shake Filtering for people with tremors
2. Voice recognition for mouse actions - CV tracking works reasonably particularly with relative control and IIR filtering and could easily be adapted to use this over the keyboard input for mouse actions.
3. Other hand segmentation methods
4. OpenCV Gesture Recognition 
