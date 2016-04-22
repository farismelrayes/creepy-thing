## Filename: facerecognition.py

# Import the juicy stuff
import cv2
import os
import numpy as np
from time import strftime
from datetime import datetime
from PIL import Image
from sys import exit
from glob import glob


# Initialise variables
IMAGES = []
LABELS = []
FACEID = []

cascadepath = 'haarcascade.xml'
facecascade = cv2.CascadeClassifier(cascadepath)
recognizer = cv2.face.createLBPHFaceRecognizer(grid_x=16, grid_y=16, threshold=1000)


# Define functions
def camera_count():
    """
    :description: finds the number of cameras connected
    """
    num = 0
    while True:
        cap = cv2.VideoCapture()
        cap.open(num)
        if cap.isOpened() == False:
            break
        else:
            num += 1
            cap.release()
    return num

def camera_seek():
    """
    :description: seeks the id of connected camera
    """
    test = True
    if camera_count() < 1:
        print("No cameras connected")
    else:
        x = 0
        while test and x < 256:
            try:
                print("Found camera with id: " + str(x))
                video_loop(x)
                test = False
            except Exception:
                x += 1
        if x > 255:
            print("No cameras in range")
    return x

def face_crop(image, remove):
    """
    :description: crops photo into images of individual faces
    """
    img = cv2.imread(image)
    pth = '\\'.join(image.split('\\')[:-1])
    fnm = image.split('\\')[-1]

    if 'facecrop_' not in fnm:
        facedata = "haarcascade.xml"
        cascade = cv2.CascadeClassifier(facedata)
        minisize = (img.shape[1], img.shape[0])
        miniframe = cv2.resize(img, minisize)
        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [v for v in f]
            subface = img[max(y-8,0):y+h+8, max(x-8,0):x+w+8]
            phototime = datetime.now()
            face_file_name = "facecrop_" + str(y) + phototime.strftime('_%Y%m%d_%H%M%S') + ".jpg"
            if len(pth) > 0:
                cv2.imwrite(pth + '\\' + face_file_name, subface)
            else:
                cv2.imwrite(face_file_name, subface)

        if remove == True:
            os.remove(image)

        return True
    return False

def crop_folder(folder):
    """
    :description: walks through folder and crops all photos to faces
    """
    for folder in os.walk(folder):
        for filetype in ['*.jpg', '*.png', '*.jpeg']:
            for file in glob(folder[0]+'\\'+filetype):
                if face_crop(file, True):
                    print(file)

def update_database(filename):
    """
    :description: learn faces from pictures and save them to a file
    :param filename: filename
    """
    i = 0
    for folder in os.walk('facedata'):
        currentperson = folder[0].split('\\')[-1]
        if currentperson != 'facedata' and currentperson != 'Unsorted':
            FACEID.append(i)
            FACEID.append(currentperson)
            print("Scanning " + currentperson + "...")
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

def get_database(filename):
    """
    :description: load faces from file
    :param filename: name of the file to load faces from
    """
    i = 0
    for folder in os.walk('facedata'):
        currentperson = folder[0].split('\\')[-1]
        if currentperson != 'facedata' and currentperson != 'Unsorted':
            FACEID.append(i)
            FACEID.append(currentperson)
            i += 1
    recognizer.load(filename)

def video_loop(camera):
    """
    :description: run camera and look for faces
    :param camera: id of the camera to use
    """
    video = cv2.VideoCapture(camera)

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            recognize_image_pil = Image.fromarray(frame).convert('L')
            recognize_image = np.asarray(recognize_image_pil, 'uint8')
            person_predicted = recognizer.predict(recognize_image[y: y + h, x: x + w])
            cv2.putText(frame, FACEID[FACEID.index(person_predicted)+1], (x, y-8), 1, 1, (255, 255, 255))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Break code
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #face_crop('people.jpg', False)
    #crop_folder('cropping')
    phototime = datetime.now()
    update_database('faces' + phototime.strftime('%Y%m%d') + '.yaml')
    #get_database('faces_20160803.yaml')
    video_loop(0)
    print("Done")
