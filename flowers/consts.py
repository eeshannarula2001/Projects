import os
import numpy as np

DIM1 = 98
DIM2 = 98
streams = 3

SHAPE_single = shape_single = (DIM1,DIM2)
SHAPE_streamed = shape_streamed = input_shape = (DIM1,DIM2,streams)

shape_streamed_one = (1,DIM1,DIM2,streams)

CATS = CAT = [ 'daisy',
               'dandelion',
               'rose',
               'sunflower',
               'tulip' ]

classes = len(CATS)

prepath = os.path.join(os.getcwd(),'data')

PATHS = [os.path.join(prepath,CAT) for CAT in CATS]

epochs = 25
lr = 0.0009925
activation = 'softmax'
loss = 'categorical_crossentropy'
