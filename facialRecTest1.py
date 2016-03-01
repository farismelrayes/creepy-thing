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
import json
import glob


# Define lists and variables
IMAGES = []
LABELS = []
FACEID = []

cascadepath = 'haarcascade_frontalface_default.xml'

facecascade = cv2.CascadeClassifier(cascadepath)
recognizer = cv2.face.createLBPHFaceRecognizer()


# Learn faces from pictures in database
i = 0
for folder in os.walk('database'):
    currentPerson = folder[0].split('\\')[-1]
    if (currentPerson != 'database'):
        FACEID.append(i)
        FACEID.append(currentPerson)
        for filetype in ['*.jpg', '*.png', '*.jpeg']:
            for file in glob.glob(folder[0]+'\\'+filetype):
                image_pil = Image.open(file).convert('L')
                image = np.array(image_pil, 'uint8')
                faces = facecascade.detectMultiScale(image)
                if len(faces) != 1:
                    os.remove(file)
                    print('Removed', file)
                else:
                    for (x, y, w, h) in faces:
                        IMAGES.append(image[y: y + h, x: x + w])
                        LABELS.append(i)
        i += 1
        print('Gathered files on: ' + currentPerson)

# Train the recognizer
recognizer.train(IMAGES, np.array(LABELS))

# Get video from webcam and run live face detection
video = cv2.VideoCapture(0)
mainloop = True

while mainloop:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        recognize_image_pil = Image.fromarray(frame).convert('L')#.crop((x, y, x+w, y+h))#.convert('L')
        recognize_image = np.array(recognize_image_pil, 'uint8')

        person_predicted = recognizer.predict(recognize_image[y: y + h, x: x + w])#, confidence
        cv2.putText(frame, FACEID[FACEID.index(person_predicted)+1], (x, y), 1, 1, (255, 255, 255))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        mainloop = False

# Exit code
video.release()
cv2.destroyAllWindows()