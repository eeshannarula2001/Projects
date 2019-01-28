import os
import cv2
import consts
import numpy as np
from sklearn.model_selection import train_test_split

def loadImages(folderpath):
    images = []
    for imgname in os.listdir(folderpath):
        path = os.path.join(folderpath,imgname)
        images.append(cv2.resize(cv2.imread(path),consts.SHAPE_single))
    return images

def loadAll():
    return [loadImages(path) for path in consts.PATHS]

def loadData():
    images = loadAll()
    X = []
    Y = []
    for l in images:
        X += l
        label = images.index(l)
        target_count = len(l)

        target = [0] * consts.classes
        target[label] = 1

        Y += [target] * target_count

    return np.array(X)/255,np.array(Y)

def get_training_testing_data():
    X,Y = loadData()
    return train_test_split(X,Y,shuffle = True)

def FromOneHot(OneHotEncodedArray):
    normal_array = []
    for target in OneHotEncodedArray:
        normal_array.append(list(target).index(max(list(target))))
    return np.array(normal_array)


def make_read_for_input(path):
    img = cv2.resize(cv2.imread(path),consts.shape_single)
    return np.reshape(img,consts.shape_streamed_one)/255     
