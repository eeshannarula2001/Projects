import os
import numpy as np

DIM1 = DIM2 = 98
streams = 3

SHAPE = shape = (DIM1,DIM2)
SHAPE_streamed = shape_streamed = input_shape = (DIM1,DIM2,streams)

shape_streamed_one = (1,DIM1,DIM2,streams)

prepath = os.path.join(os.getcwd(),'cell_images')

CATS = [ 'Parasitized',
         'Uninfected' ]

classes = len(CATS)

PATHS = paths = [os.path.join(prepath,CAT) for CAT in CATS]

epochs = 1
lr = 0.009589
activation = 'softmax'
loss = 'categorical_crossentropy'

trained_model_name = 'malaria90.h5'
prepath_fortrainedmodel = os.path.join(os.getcwd(),'models')
trained_model = os.path.join(prepath_fortrainedmodel,trained_model_name)
