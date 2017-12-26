import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
import pickle
from tqdm import tqdm
import joblib
from keras.datasets import cifar10

# model = InceptionV3(weights='imagenet', include_top=False)

model = keras.models.load_model("./checkpoints/Iweights_vision-199-0.11.hdf5")
model.pop()

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # return model.predict(x)
    return model.predict(x)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # weights = model.predict(x_train)
    # weights = model.predict(x_train[:1])
    # joblib.dump(weights, "./weights-vision.pkl", compress=True)

    loaded_wt = joblib.load("./weights-vision.pkl")
    # result = []
    # for i in range(20):
    #     result.append(cosine_similarity(loaded_wt, model.predict(x_test[i:i+1])))
    result = (cosine_similarity(loaded_wt, model.predict(x_test[:1])))
    print result
    # print np.argsort(result)[-3:][::-1]


