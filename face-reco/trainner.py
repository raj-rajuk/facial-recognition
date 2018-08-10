import cv2
import os
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create();
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path='dataSet'
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    #create empth face list
    faces=[]
    #create empty ID list
    IDs=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        faceImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        faceNp=np.array(faceImage,'uint8')
        faces.append(faceNp)
        #getting the Id from the image
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        print(ID)
        # extract the face from the training image sample
        #If a face is there then append that in the list as well as Id of it
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return faces,IDs


faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('train/trainner.yml')
cv2.destroyAllWindows()
