import consts
import numpy as np
from data import Data_manager

data_manager = Data_manager()
data_manager.addAll(consts.paths)

X,Y = data_manager.loadData()

np.save('x',X/255)
np.save('y',Y)
