import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

#arr = np.zeros([3,3])
#arr[1,1] = 1
#arr[0, 2] = 2
#print(arr[arr!=0])
#upp_tri = arr[np.triu_indices(arr.shape[1], 0)]
#print(upp_tri)
#@values = ["a", "b", "c"]

#for i, value in enumerate(values):
    #print(i, value)

#for i in range(10):
    #for j in range(i, 10):
        #print(i, j)

arr = np.array([1, 2, 3, 4, 5])

arr = arr[arr != 5]

print(arr)