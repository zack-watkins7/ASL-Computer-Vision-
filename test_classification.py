import time
from typing import Counter
import mediapipe as mp
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

#Global Variables
cap = cv2.VideoCapture(0)
detector = HandDetector()
pastTime=0
currentTime=0
staticWindowSize=300
counter=0
folder= 'training_data\letter_C'
classifier= Classifier('Model\keras_model.h5', 'Model\labels.txt')
labels= ["A", "B", "C","D","E","F"]

class Functionality:
    def quitProgram(self):
        cap.release()
        cv2.destroyAllWindows
    def saveImage(self,folder):
        cv2.imwrite(f'{folder}/training_image_{time.time()}.jpg', whiteStaticBox)

functionality = Functionality()



while True:
    success, img= cap.read()
    hands,img= detector.findHands(img)
    imgCopy = img.copy()
    offset=20


    if hands:
        hand= hands[0]
        x,y,w,h= hand['bbox']
        whiteStaticBox= np.ones((staticWindowSize,staticWindowSize,3), dtype=np.uint8)*255
        croppedImage= img[y-offset: y+h +offset , x-offset:x+w +offset]       #img is a matrix so you need to define the appropriate coordinates.
        
        #shapeOfCropped= croppedImage.shape
        
        #making a window that only readjusts the width
        aspectRatio=h/w
        if aspectRatio > 1:
            k = staticWindowSize/h
            widthCalculation= math.ceil(k*w)
            imgResize= cv2.resize(croppedImage,(widthCalculation, staticWindowSize))
            resizeShape= imgResize.shape
            widthGap= math.ceil((staticWindowSize-widthCalculation)/2)
            whiteStaticBox[:, widthGap: widthCalculation+widthGap] = imgResize
            prediction, index= classifier.getPrediction(whiteStaticBox, draw= False)


        else:
            k = staticWindowSize/w
            heightCalculation= math.ceil(k*h)
            imgResize= cv2.resize(croppedImage,(staticWindowSize,heightCalculation))
            resizeShape= imgResize.shape
            heightGap= math.ceil((staticWindowSize-heightCalculation)/2)
            whiteStaticBox[heightGap:heightCalculation+heightGap, :] = imgResize
            prediction, index= classifier.getPrediction(whiteStaticBox, draw= False)

              
        cv2.imshow("cropped", croppedImage)
        cv2.imshow("staticBox", whiteStaticBox)
    
        cv2.putText(imgCopy, labels[index],(10,70), cv2.FONT_HERSHEY_DUPLEX,3, (255,0,255),3 )
    cv2.imshow("image", imgCopy)
    cv2.waitKey(1)

    
    if cv2.waitKey(1) == ord('q'):
        functionality.quitProgram()
    