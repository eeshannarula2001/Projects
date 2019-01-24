import os

DIM1 = 98
DIM2 = 98
streams = 3

SHAPE_single = (DIM1,DIM2)
SHAPE_streamed = (DIM1,DIM2,streams)
SHAPE_streamed_one = (1,DIM1,DIM2,streams)

shape_single = (DIM1,DIM2)
shape_streamed = (DIM1,DIM2,streams)
shape_streamed_one = (1,DIM1,DIM2,streams)

input_shape = SHAPE_streamed

lr = 0.01
epochs = 5
activation = 'softmax'
input_shape = shape_streamed
loss = 'categorical_crossentropy'

classes = 2

CAT = 0
DOG = 1

CATS = ['cats',
        'dogs']

prepath = os.path.join(os.getcwd(),'dataset')

paths = [ os.path.join(prepath,CATS[CAT]),
          os.path.join(prepath,CATS[DOG]) ]
