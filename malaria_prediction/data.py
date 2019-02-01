import numpy as np
import consts
import tools
import cv2
import os

def getData():
    images = []
    targets = []
    for path in consts.PATHS:
        label = consts.PATHS.index(path)
        target = [0] * consts.classes
        target[label] = 1
        for image in os.listdir(path):
            image_path = os.path.join(path,image)
            images.append(tools.loadImage(image_path,consts.SHAPE))
            targets.append(target)
    return tools.train_test(np.array(images)/255.0,np.array(targets),True)
