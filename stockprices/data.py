# constents
import consts

# for matrix math
import numpy as np

# for the stock data
from yahoo_fin.stock_info import *

# function to load data
def getdata():

    '''
    firt we will get the data from the yahoo api.
    the api has a function called 'get_data' which
    takes in the name of the company whoes stocks we
    want.
    '''

    data = get_data(consts.company)

    '''
    Now there are 4 kind of prices : 'open','high', 'low' and 'close'.
    We want to take out the average of all the prices to get one price.
    '''

    array = []

    for feature in consts.features:
        array.append(np.array(data[feature]))

    sum = np.zeros(array[0].shape)

    for i in array:
        sum += i

    avg = sum/4

    return avg
