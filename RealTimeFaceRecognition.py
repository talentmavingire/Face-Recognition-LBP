#!/usr/bin/env python
# coding: utf-8

# #### Testing the Camera and running the video

# In[14]:


import glob
from PIL import Image
import os
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


# #### 1. Detecting a face using haarcascade
# #### 2. Forming a square around any face detected.

# In[17]:


# videoFile = "Class Activity.mp4" # Capture faces from video
cap = cv2.VideoCapture(0)  # replace argument with videoFile variable
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(videoFile)
cap.set(1, 640)  # set Width
cap.set(1, 480)  # set Height
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()


# #### Enter id for every different face and gather images of faces.

# In[8]:


# videoFile = "Class Activity.mp4" # Capture faces from video
cap = cv2.VideoCapture(0)  # replace argument with videoFile variable
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cap.read()
    # img = cv2.flip(img, -1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
        # C:\Users\mavin\Desktop\FaceRecognition\dataset\train

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/train/User" +
                    str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Camera', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 10:  # Take 10 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()


# #### Train on the basis of gathered data
# #### Here I am using Local Binary Pattern Histograms Algorithm

# In[18]:


# Path for face image database
path = np.sort(glob.glob('dataset/train/User*.jpg'))
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    faceSamples = []
    faceLabels = []
    # iterate through the file path and get each image
    for i in path:
        img = cv2.imread(i, 0)
        # print(i)
        faceLabel = int(i.split('.')[1])
        faces = detector.detectMultiScale(img)
        for (x, y, w, h) in faces:
            faceSamples.append(img[y:y+h, x:x+w])
            faceLabels.append(faceLabel)

    return faceSamples, faceLabels


faces, labels = getImagesAndLabels(path)
labels = np.array(labels)
labels.astype(int)
print(labels)
recognizer.train(faces, np.array(labels))
recognizer.write('dataset/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(
    len(np.unique(labels))))


# #### Providing a different video section and recognizing the actor by name

# In[11]:


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('dataset/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0
# names related to ids: example ==> Paul: id=1,  etc
names = ['None', 'Paul', 'Paul', 'Paul', 'Paul',
         'Paul', 'Conrad', "Conrad", 'Conrad', "Conrad", 'Conrad']


# Initialize and start realtime video capture
# videoFile = "Class Activity.mp4"
# cam = cv2.VideoCapture(videoFile)

# #live feed
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(int(minW), int(minH)),
    )
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (0, 255, 0), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5),
                    font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


# In[ ]:
