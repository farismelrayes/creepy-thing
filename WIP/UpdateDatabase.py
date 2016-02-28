import cv2, os
import numpy as np
import pickle
import dill
from PIL import Image

#Things that make things work
arguments = ['haarcascade_frontalface_default.xml']
peopleIDs = []
idElement = 0

# Get user supplied values
cascPath = arguments[0]
faceCascade = cv2.CascadeClassifier(cascPath)

#Creates the face recogniser
recognizer = cv2.face.createLBPHFaceRecognizer()
peopleIDs = []
idElement = 0


#Teaches the recognizer based on images in the database
images = []
labels = []
for folder in os.walk('Database'):
    currentPerson = folder[0].split("\\")[-1]
    if (currentPerson!='Database'):#Nobody in the school can be named database now or else
        peopleIDs.append(idElement)#Also hopefully nobody's name is a number
        peopleIDs.append(currentPerson)
        for file in folder[2]:
            #Check and see if a face is detected in the image
            #First convert the image to grayscale
            image_pil = Image.open('Database\\'+currentPerson+'\\'+file).convert('L')
            image = np.array(image_pil, 'uint8')
            #Detect it
            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(idElement)
        idElement+=1
        print('Gathered files on '+currentPerson)

#Trains the recognizer
recognizer.train(images,np.array(labels))

for attribute in dir(recognizer):
    print(attribute)
    print(getattr(recognizer, attribute))

#Saves recognizer and ids to pickles
dill.dump(recognizer, open('recognizer.p','w'))



#with open('recognizer.p', 'w') as handle:
#    dill.dump(recognizer, handle)

with open('peopleids.p', 'wb') as handle:
    pickle.dump(peopleIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Updated, good looking!")