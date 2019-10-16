import cv2
import sys
import numpy as np

print(cv2.__version__)

capture = cv2.VideoCapture(1)
ret, initial_frame = capture.read()
i_frame_g = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = capture.read()
    frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame is None:
        break
    
    #Do background subtraction
    fgMask = frame_g - i_frame_g
    _, fgMask_T = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)
    
    #Perform Morphology to remove noise and fill holes
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(fgMask_T, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('FG Mask', fgMask_T)
    cv2.imshow('Post Morphology', closing)
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv2.destroyAllWindows()