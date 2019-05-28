import data
import model
import tools
import consts

def getmodel():

    X,Y,theta = data.getData()
    return model.train(X,Y,theta,consts.alpha,consts.epochs,True)



print(getmodel())
