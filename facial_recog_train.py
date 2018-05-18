
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image
import cv2


# In[3]:


recog = cv2.createLBPHFaceRecognizer();
path = 'dataset'

def getImageswithID(path):
    imagePaths =[os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('_')[0])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids,faces = getImageswithID(path)
recog.train(faces,np.array(Ids))
recog.save('trainingData.yml')
cv2.destroyAllWindows()
