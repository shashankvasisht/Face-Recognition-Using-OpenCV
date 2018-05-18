import cv2
import numpy as np



facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.createLBPHFaceRecognizer();
rec.load('trainingData.yml')

id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)

id_map = ['Shashank','Srishti','John Mayer','Big B']


cam = cv2.VideoCapture(0)
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+h])
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id_map[id-1]),(x,y+h),font,255);
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()

