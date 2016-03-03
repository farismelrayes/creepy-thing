import cv2

def facecrop(image):
    facedata = "haarcascade.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)
    
    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    i = 0
    for f in faces:
        i += 1
        x, y, w, h = [v for v in f]
        #cv2.rectangle(img, (x-9,y-9), (x+w+8,y+h+8), (255,255,255))
        
        subface = img[max(y-8,0):y+h+8, max(x-8,0):x+w+8]
        face_file_name = "Database/face_" + str(i) + ".jpg"
        cv2.imwrite(face_file_name, subface)

    cv2.imshow(image, img)

    return
 
if __name__ == '__main__':
    facecrop('people.jpg')
    
    while(True):
        key = cv2.waitKey(20)
        if key in [27, ord('Q'), ord('q')]:
            break