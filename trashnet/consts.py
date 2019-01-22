import os


GLASS = 0
PAPER = 1
CARDBOARD = 2
PLASTIC = 3
METAL = 4


CAT = [
'glass',
'paper',
'cardboard',
'plastic',
'metal',
'trash'
]

DIM1 = 98
DIM2 = 98
streams = 3

shape_single = (DIM1,DIM2)
shape_streamed = (DIM1,DIM2,streams)
shape_streamed_one = (1,DIM1,DIM2,streams)

classes = 5


prepath = os.path.join(os.getcwd(),'dataset-resized')

paths = [   os.path.join(prepath,CAT[GLASS]),
			os.path.join(prepath,CAT[PAPER]),
			os.path.join(prepath,CAT[CARDBOARD]),
			os.path.join(prepath,CAT[PLASTIC]),
			os.path.join(prepath,CAT[METAL]) 
		]


lr = 0.0009925
epochs = 20
activation = 'softmax' 			
input_shape = shape_streamed
loss = 'categorical_crossentropy'
