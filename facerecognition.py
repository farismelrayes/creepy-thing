## Filename: facerecognition.py
# Import the juicy stuff
import cv2
import os
import numpy as np
from PIL import Image
from sys import exit
from glob import glob


# Define lists and variables
IMAGES = []
LABELS = []
FACEID = []

cascadepath = 'haarcascade.xml'
facecascade = cv2.CascadeClassifier(cascadepath)
recognizer = cv2.face.createLBPHFaceRecognizer()


# Crop photo into faces
def facecrop(image):
    facedata = "haarcascade.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]

        subface = img[max(y-8,0):y+h+8, max(x-8,0):x+w+8]
        face_file_name = "faces/face_" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, subface)

# Learn faces from pictures, save to file
def updatedatabase(filename):
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
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            IMAGES.append(image[y: y + h, x: x + w])
                            LABELS.append(i)
                    else:
                        os.remove(file)
            i += 1
            
    recognizer.train(IMAGES, np.array(LABELS))
    recognizer.save(filename)

# Load faces from file
def getdatabase(filename):
    i = 0
    for folder in os.walk('database'):
        currentPerson = folder[0].split('\\')[-1]
        if (currentPerson != 'database'):
            FACEID.append(i)
            FACEID.append(currentPerson)
            i += 1
    recognizer.load(filname)

# Live video face recognition
def videoloop():
    video = cv2.VideoCapture(0) # USB Cam = 0; Laptop Cam = 1;
    while 1:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            recognize_image_pil = Image.fromarray(frame).convert('L')#.crop((x, y, x+w, y+h))#.convert('L')
            recognize_image = np.array(recognize_image_pil, 'uint8')
            person_predicted, confidence = recognizer.predict(recognize_image[y: y + h, x: x + w])#, confidence
            cv2.putText(frame, FACEID[FACEID.index(person_predicted)+1], (x, y-8), 1, 1, (255, 255, 255))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Break code
            break
    video.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    getdatabase('facesavetest.yaml')
    videoloop()