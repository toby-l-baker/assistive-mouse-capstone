# Camera Mouse Folder for Development

## Contents
1. camera.py: where to create camera objects for various cameras (currently implemented are webcams and the real sense)
2. cameramouse.py: camera mouse implementations bringing together gesture recognition, trackers, camera and OS
3. interface.py: contains objects specific to different OS such as a windows mouse and windows monitor
4. tracker.py:
- where you create instances of your tracker
- Should output the normalized velocity of the hand
- Will typically take the input of new frames or the output of mediapipe
5. main.py: where you run your camera mouse
6. gesture_recognition.py:
- the place to store gesture recognition objects
- should alter the Gesture object of the camera mouse to reflect the current gesture
- will likely take in images as the input or take in the output of mediapipe

<img src="Untitled Diagram.png"/>
