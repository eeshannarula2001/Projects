#gather all the libraries
import consts
from data import getdata
from model import makeModel

# Now we wannt numpy for martix math operations
import numpy as np

# for making the grap we will import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# geting the data
data = getdata()

#making the model
model = makeModel(consts.lr,consts.th)

# initialzing variables for graph
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

'''
Now we will be training the model with the live stock price graph.
we will make a function named animate which will be the function
which will interate with time.

this function has a variable 'i' which will be the number of interation
done.

Then we will make a function which will take the number of interations done
and output us the training data and label.

lets just first do this
'''


def animate(i): # where 'i' is no. of interation done

    # we will make an array of the data it has reached till now
    array = data[0:i]

    '''
    when the value of i is greater than threshold + 1
    we can allow the model to get trained
    '''

    if i > consts.threshold + 1:

        startTraining(array,i)   # we will write this function ...

        '''
        now we will plot the points on the graph
        and the bought and sell points in red and
        blue respectively
        '''

        # first we need to clear the graph
        ax1.clear()

        # plot the graph
        ax1.plot(np.array(range(0,array.shape[0])),array)

        # plot the sell buy points..

        # first buy points with red color ('ro' indicates red dot)
        ax1.plot(consts.x_buyingprice,consts.y_buyingprice,'ro')

        # then blue colored selling points ('bo' indicates blue dot)
        ax1.plot(consts.x_sellingprice,consts.y_sellingprice,'bo')

'''
now we will write a function which will train our model
and predict if we will buy or sell our stocks

it takes in the current data array and the current interation number
'''
def startTraining(array,i):
    input,output = getinputoutput(array,i) # we will write this function

    '''
    now we will take a prediction of the input
    then we will train it on the input and correct
    answer.
    '''

    prediction = model.predict(input)[0][0]

    '''
    the prediction variable is between 0 and 1.
    as discussed befour this value aproches 1 for
    increase in price and aproches 0 for decrease in
    price.

    so we will set a threshold value for buying and selling
    the stocks. i have set the values of selling_threshold and
    buying_threshold to .54 and .46 respectively.

    i have written a seperate function for it
    now i will just call it (updateStatus)

    it will take out prediction and tell if we should
    buy or sell or donothing
    '''

    updateStatus(array,prediction,i) # we will write this function

    '''
    now the only thing left is to train our model
    with the input and output we defined earlyer
    '''

    model.fit(input,output,epochs = consts.epochs,verbose=0)

'''
Now we will write the getinputoutput function which
we used in the function startTraining to get input and output

it will take the same parameters as startTraining function
'''
def getinputoutput(array,i):

    # so our input would be the last threshold amout of entries
    input = np.array([array[i - (consts.threshold + 1) : i - 1]])

    '''
    now lets talk about how our output should look.
    As it is a classification problem where if output
    is 1 there is increase in price and if output is zero
    then there is dicrease in price.

    so now from out prices we will find if there is a gain
    or loss and assign value to the output accordingly
    '''

    output = 0

    if array[i - 1] > array[i - 2]: # here a[i - 2] is the last value and a[i - 1] is current value
        output = 1

    output = np.array([output])

    return input,output

'''
this is the function we called in the function
startTraining at line 107.

this function will buy and sell on the bases of
the prediction made by our model
'''

def updateStatus(array,prediction,i):

     '''
     if the predictide value is greater than selling_threshold
     then we will sell stocks and if it is less than buying_threshold
     then we will buy stocks
     '''

     if prediction < consts.buying_threshold:
         '''
         now if we have bought somthing befour then
         we will not allow the model to buy another stock.
         '''
         if len(consts.y_buyingprice) == len(consts.y_sellingprice):

             '''
             now we can buy stocks

             what this if condition is checking is that
             is there some stocks which not sold yet. if
             all the stocks are sold then we can buy new
             stocks.
             '''

             consts.x_buyingprice.append(i)
             consts.y_buyingprice.append(array[i-1])

     if prediction > consts.selling_threshold:

         '''
         now first we need to check if we have bought some stocks.
         if we havent bought stocks the are we going to sell stocks.

         its like you can not eat a candy without buying it.
         '''

         if len(consts.y_buyingprice) != len(consts.y_sellingprice):

             '''
             now what this if condition has checked is that is there
             any stocks which are not sold yet.if their are then we can
             continue to sell stocks
             '''

             consts.x_sellingprice.append(i)
             consts.y_sellingprice.append(array[i-1])

'''
Now is the last step

we will put our interative function, that is
animate into the built-in function animation
for live graph
'''

ani = animation.FuncAnimation(fig, animate, interval = consts.interval)
plt.show()
