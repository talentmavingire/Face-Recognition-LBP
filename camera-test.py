import numpy as np
import cv2
videoFile = "Class Activity.mp4"
cap = cv2.VideoCapture(videoFile)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
while(True):
    ret, frame = cap.read()
    # frame = cv2.flip(frame, -1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
   # cv2.imshow('gray', gray)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
