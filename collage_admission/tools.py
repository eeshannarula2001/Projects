import numpy
import consts
import pandas
from sklearn import preprocessing


def getData(df):

    fileds = list(df.columns)
    X = numpy.array(normalize_data(df.iloc[:,1:len(fileds) - 1]))
    # X = numpy.array(df.iloc[:,1:len(fileds) - 1])
    Y = numpy.array(df.iloc[:,len(fileds) - 1])
    return X,Y

def normalize_data(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pandas.DataFrame(x_scaled)
