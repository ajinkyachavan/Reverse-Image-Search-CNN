import cv2
import numpy as np
import os
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
from scipy import fftpack
import cv2
import os

RESIZE_W = 32
RESIZE_H = 32

np.set_printoptions(threshold=np.nan)

'''
Reading data
'''
print("\n\n")
print("Applying SIFT...")

def load(filename):
 
  with open(filename, 'rb') as f:
    if sys.version_info[0] < 3:
      dict = pickle.load(f)
    else:
      dict = pickle.load(f, encoding='latin1')
    x = dict['data']
    y = dict['labels']
    x = x.astype(float)
    y = np.array(y)
  return x, y

Xtrain = []
ytrain = []

for i in range(1):
  a,b = (load("cifar_10/data_batch_"+str(i+1)))
  Xtrain.append(a)
  ytrain.append(b)

Xtrain_data = np.array(Xtrain)
ytrain_data = np.array(ytrain)


Xtest, ytest = load("cifar_10/test_batch")

Xtest = np.array(Xtest)
ytest = np.array(ytest)

#60000 images of 32x32 = 50000 training images of 32x32x3 + 10000 test images of 32x32x3

#print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
#(5, 10000, 3072) (5, 10000) (10000, 3072) (10000,)
# ytrain[0].shape = 10000 => output between 0 to 9 i.e for each class of 10000 images of 3072 shape

Xtrain = []
ytrain = []



Xtrain = np.reshape(Xtrain_data, (10000, 3072))
Xtrain = Xtrain[:500]
ytrain = ytrain_data.ravel()


Xtest = np.reshape(Xtest, (10000,3072))
Xtest = Xtest[:100]
ytest = ytest.ravel()


x,y = Xtest.shape

sift = cv2.xfeatures2d.SIFT_create()

dist = []

for i in range(x):
  for j in range(x):

    img1 = cv2.imread('test/Xtest'+str(i)+'.jpg')
    #img1 = cv2.imread('car1.jpg')

    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1,None)

    img2 = cv2.imread('test/Xtest'+str(j)+'.jpg')
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    maxVal = min(len(kp1), len(kp2))

    distMat = []

    for d in range(maxVal):
      
      distMat.append(( (kp1[d].pt[0]-kp2[d].pt[0])**2 + (kp1[d].pt[1]-kp2[d].pt[1])**2 ))

    dist.append(np.sum(distMat))

dist = np.array(dist)
dist = np.reshape(dist, (100,100))

#print(dist)

####Testing
print("Testing...")

for i in range(x):
  for j in range(x):
    if(dist[i][j] < 2500):
      dist[i][j] = ytest[i]
    else:
      dist[i][j] = -1


accuracy = 0

for i in range(x):
  for j in range(x):
    if(ytest[i] - dist[i][j]):
      accuracy += 1

print("Accuracy - ",accuracy/(x**2)*100)

