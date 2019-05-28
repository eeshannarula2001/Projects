import tools
import numpy
import model
import consts
import pandas


def getData():
    df = pandas.read_csv(consts.filename)

    # print(df.head())

    X,Y = tools.getData(df)
    theta = numpy.zeros(X.shape[1]+1)

    return X,Y,theta
    # print(model.train(0.1,15,X,Y,theta))
