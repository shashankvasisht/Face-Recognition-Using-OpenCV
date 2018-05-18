
# coding: utf-8

# In[2]:


import cv2


# In[3]:


import numpy as np
#from cv2 import CascadeClassifier


# In[5]:


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[6]:


cam = cv2.VideoCapture(0)
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()

