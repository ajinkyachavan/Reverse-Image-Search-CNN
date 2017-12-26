
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

from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import  RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

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

import time


start = time.clock()
'''
Reading data
'''

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
Xtrain = Xtrain[:1000]
ytrain = ytrain_data.ravel()
ytrain = ytrain[:1000]

Xtest = np.reshape(Xtest, (10000,3072))
Xtest = Xtest[:100]
ytest = ytest.ravel()
ytest = ytest[:100]




class PA1:

    def __init__(self, estimator, Xtrain, ytrain, Xtest, ytest):
        #self.data, self.label = self.preprocess_data(estimator)
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.estimator = estimator    



    def train(self, n_examples=None):

        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        print("fitting")
        self.estimator.fit(self.Xtrain, self.ytrain)
        print("predicting")
        y_pred = self.estimator.predict(self.Xtest)
        

        print( classification_report(ytest, y_pred))

        
if __name__ == "__main__":

    
    seed = np.random.randint(100000)


    estimator = SVC(C=1.0, probability=True)
    

    pa1 = PA1(estimator, Xtrain, ytrain, Xtest, ytest)    
    
    pa1.train()

print("Time taken = ",str(start-time.clock()))