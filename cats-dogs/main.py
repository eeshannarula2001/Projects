import os
import cv2
import consts
import numpy as np
from data import Data_manager
from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.models import Sequential


X,Y = np.load('x.npy'),np.load('y.npy')
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle = True)


model = Sequential()

model.add(Conv2D(kernel_size = (3,3),filters = 32,activation = 'tanh',input_shape = consts.input_shape))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(kernel_size = (3,3),filters = 64,activation = 'tanh'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(100, activation = 'tanh'))
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(16, activation = 'tanh'))

model.add(Dense(consts.classes,activation = consts.activation))

optimizer = SGD(lr = consts.lr,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer,loss = consts.loss)

## training the model
model.fit(x_train,y_train,epochs=consts.epochs,shuffle=True,validation_data=(x_test,y_test))

prediction = Data_manager.FromOneHot(model.predict(x_test))
correct = Data_manager.FromOneHot(y_test)


acc = str((np.sum(prediction == correct)/len(list(y_test))) * 100)
print('Acc :' + acc + '%')

model.save('cats-dogs_net.h5')