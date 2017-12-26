from PIL import Image
import imagehash
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
from scipy import fftpack
import cv2


RESIZE_W = 32
RESIZE_H = 32

np.set_printoptions(threshold=np.nan)



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


Xtest = np.reshape(Xtest, (10000,3072))
Xtest = Xtest[:1000]
ytest = ytest.ravel()


import os
import re

non_decimal = re.compile(r'[^\d]+')

x,y = Xtest.shape



'''
## Uncomment this block For training


for i in range(x):
	f = open("outputHash/output_"+str(i)+".txt","w")

	for j in range(x):

		hash1 = imagehash.average_hash(Image.open('test/Xtest'+str(i)+'.jpg'))
		#print(hash1)

		hash2 = imagehash.average_hash(Image.open('test/Xtest'+str(j)+'.jpg'))
		#print(hash2)

		f.write(str(i)+","+str(j)+","+str(math.fabs(hash1-hash2))+"\n")
		#print(hash1 - hash2)

	f.close()	
'''	


### Uncomment this For testing 

pHashProba = []

for i in range(x):
	f = open("outputHash/output_"+str(i)+".txt")


	for row in f:
		first, second, pHash = row.split(",")

		if(float(pHash) < 50):
			pHash = ytest[i]
		else:
			pHash = -1

		pHashProba.append(pHash)


#checking accuracy
accuracy = 0

for i in range(x):
	if(ytest[i] == pHashProba[i]):
		accuracy += 1

print(accuracy/x *100 )


