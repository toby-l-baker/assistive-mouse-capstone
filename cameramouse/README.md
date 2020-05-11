# OpenCV Only Full Python  Implementation

### Files of interest to run 

Make sure to edit config.yaml to match your setup and the different control/segmentation type you want to use
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

### Future Work

1. Anti-Shake Filtering for people with tremors
2. Voice recognition for mouse actions - CV tracking works reasonably particularly with relative control and IIR filtering and could easily be adapted to use this over the keyboard input for mouse actions.
3. Other hand segmentation methods
4. OpenCV Gesture Recognition 
