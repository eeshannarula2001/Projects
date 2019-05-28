import matplotlib.pyplot as plt
import numpy
import math

def cost(x,y,theta):
    # theta must have feature_count + 1 values

    X = numpy.insert(x,0,1,axis = 1)

    m = X.shape[0]
    H = numpy.matmul(X,theta)
    J = (1/(2*m)) * numpy.sum(numpy.square(H - y))

    grad = []
    for i in range(X.shape[1]):
        grad.append((1/m) * numpy.sum(numpy.dot((H - y),X[:,i])))

    return J,grad

def predict(X,theta):
    return numpy.matmul(X,theta)

def train(X,Y,theta,alpha,epochs,notgraph):
    c = []
    for i in  range(epochs):
        J,grad = cost(X,Y,theta)
        delta  = alpha * numpy.array(grad)
        theta = theta - delta
        c.append(J)
        print('cost:' + str(J))
    if notgraph == False:
        plt.plot(range(0,len(c)),c)
        plt.show()
    return theta    
