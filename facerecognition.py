"""

TODO:
-Clean up code
-Crop photos to face
-Saving/loading

"""


# Import the juicy stuff
import cv2
import os
import numpy as np
from PIL import Image
from sys import exit


# Define lists and variables
FACEID = []

cascadepath = 'haarcascade.xml'
facecascade = cv2.CascadeClassifier(cascadepath)
recognizer = cv2.face.createLBPHFaceRecognizer()


# Load faces
i = 0
for folder in os.walk('database'):
    currentPerson = folder[0].split('\\')[-1]
    if (currentPerson != 'database'):
        FACEID.append(i)
        FACEID.append(currentPerson)
        i += 1
try:
    recognizer.load('facesavetest.yaml')
except:
    print("\nFile 'facesavetest.yaml' cannot be found.\n")
    raise SystemExit


# Get video from webcam and run live face detection
video = cv2.VideoCapture(0) # USB Cam = 0; Laptop Cam = 1;
mainloop = True

while mainloop:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw onto the video
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        recognize_image_pil = Image.fromarray(frame).convert('L')#.crop((x, y, x+w, y+h))#.convert('L')
        recognize_image = np.array(recognize_image_pil, 'uint8')

        person_predicted = recognizer.predict(recognize_image[y: y + h, x: x + w])#, confidence
        cv2.putText(frame, FACEID[FACEID.index(person_predicted)+1], (x, y), 1, 1, (255, 255, 255))

    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        mainloop = False


# Exit code
video.release()
cv2.destroyAllWindows()