## Filename: updatefacemaps.py
# Import the juicy stuff
import cv2
import os
import numpy as np
from PIL import Image
from glob import glob


# Define lists and variables
IMAGES = []
LABELS = []
FACEID = []

cascadepath = 'haarcascade.xml'
facecascade = cv2.CascadeClassifier(cascadepath)
recognizer = cv2.face.createLBPHFaceRecognizer()


# Learn faces from pictures in database
i = 0
for folder in os.walk('database'):
    currentPerson = folder[0].split('\\')[-1]
    if (currentPerson != 'database'):
        FACEID.append(i)
        FACEID.append(currentPerson)
        print("Scanning " + currentPerson + "...")
        for filetype in ['*.jpg', '*.png', '*.jpeg']:
            for file in glob(folder[0]+'\\'+filetype):
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

print("\nTraining system...")
recognizer.train(IMAGES, np.array(LABELS))

print("Saving data...")
recognizer.save('facesavetest.yaml')

print("\nDONE\n")