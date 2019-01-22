import numpy as np
import cv2
import os
import consts


class Data:
    def __init__(self):
        self.classes = consts.classes
        self.paths = []
    def add(self,path):
       self.append(path)

    def addAll(self,paths_list):
        for path in paths_list:
            self.paths.append(path)

    def loadData(self):
        data = []
        targets = []
        for path in self.paths:
            imgs = self.load_pics(path,consts.shape_single)
            label = self.paths.index(path)
            for img in imgs:
                data.append(img)
                target = [0] * consts.classes
                target[label] = 1
                targets.append(target)

        return np.array(data),np.array(targets)  

 
    def load_pics(self,foldername,shape):
        images = []
        for imagename in os.listdir(foldername):
            path = os.path.join(foldername,imagename)
            
            image = cv2.resize(cv2.imread(path),shape)
            images.append(image)

        return images
    
    @staticmethod    
    def FromOneHot(OneHotEncodedArray):
        normal_array = []
        for target in OneHotEncodedArray:
            normal_array.append(list(target).index(max(list(target))))
        return np.array(normal_array)

    @staticmethod
    def make_read_for_input(path):
        img = cv2.resize(cv2.imread(path),consts.shape_single)
        return np.reshape(img,consts.shape_streamed_one)/255

        

