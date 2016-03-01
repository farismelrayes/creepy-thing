"""

Currently testing different things with the saving

TODO:
-Clean up code
-Query all photos with two faces
-Crop photos to face
-Etc

"""

import cv2
import os
import numpy as np
from PIL import Image
import json
import glob

#Things that make things work
arguments = ['haarcascade_frontalface_default.xml']
IMAGES = []
LABELS = []
FACEID = []
idElement = 0

# Get user supplied values
cascPath = arguments[0]
faceCascade = cv2.CascadeClassifier(cascPath)

#Creates the face recogniser
recognizer = cv2.face.createLBPHFaceRecognizer()
idElement = 0

# Learn faces from pictures in database
for folder in os.walk('database'):
    currentPerson = folder[0].split('\\')[-1]
    if (currentPerson != 'database'):
        FACEID.append(idElement)
        FACEID.append(currentPerson)
        for filetype in ['*.jpg', '*.png', '*.jpeg']:
            for file in glob.glob(folder[0]+'\\'+filetype):
                image_pil = Image.open(file).convert('L')
                image = np.array(image_pil, 'uint8')
                faces = faceCascade.detectMultiScale(image)
                if len(faces) != 1:
                    os.remove(file)
                    print('Removed', file)
                else:
                    for (x, y, w, h) in faces:
                        IMAGES.append(image[y: y + h, x: x + w])
                        LABELS.append(idElement)
        idElement += 1
        print('Gathered files on: ' + currentPerson)

#Trains the recognizer
recognizer.train(IMAGES, np.array(LABELS))

#Things that make things work
arguments = ['haarcascade_frontalface_default.xml']

# Get user supplied values
cascPath = arguments[0]
faceCascade = cv2.CascadeClassifier(cascPath)

#Get the video thingy
video_capture = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #Make it all gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces in the image
    faces = faceCascade.detectMultiScale (gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #See if there's anyone we recognize in the rectangle
        recognize_image_pil = Image.fromarray(frame).convert('L')#.crop((x, y, x+w, y+h))#.convert('L')
        recognize_image = np.array(recognize_image_pil, 'uint8')

        person_predicted = recognizer.predict(recognize_image[y: y + h, x: x + w])#, confidence
        cv2.putText(frame, FACEID[FACEID.index(person_predicted)+1], (x, y), 1, 1, (255, 255, 255))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #Exit the script if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()