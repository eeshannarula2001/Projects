# constents
import consts

# for matrix math
import numpy as np

# for the stock data
from yahoo_fin.stock_info import *

# function to load data
# it will take in the percentage of data we want it to return
# this is because the data is too large and we dont want that much of data
def getdata(percentdata):

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

    '''
    Now we only want some percent of the data
    '''

    length = round(avg.shape[0] * percentdata * 0.01)

    # now we will reverse the array any take out some percent of data
    avg = np.flip(np.flip(avg)[0:length])

    return avg
