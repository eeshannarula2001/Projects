# keras for making the model
import keras
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.models import Sequential

'''
We will be making a 3 layer fully conected nural network
with sigmoid activation function and 1 node in the output layer.

output layer has 1node because it is a classification problem.
it will aproch 1 for increase in price and 0 if price decrease.

and...

we will be using SGd optimizer with the mean_squared_error loss fuction
'''

# function for making the model.
# it will take two parameters that are learning rate and the threshold

'''
Now what is this threshold?
it is the number previous prices we will take as our input
'''

def makeModel(lr,th):

    model = Sequential()

    model.add(Dense(th,activation = 'sigmoid'))
    model.add(Dense(1,activation = 'sigmoid'))

    optimizer = SGD(lr)
    model.compile(optimizer, loss='mean_squared_error', metrics=["accuracy"])

    return model
