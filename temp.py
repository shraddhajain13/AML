#import numpy as np
#A = np.array([[ 4,  0,  3],[ 2,  4, -2],[-2, -3,  7]])
#upp_A = A[np.triu_indices(3, 1)]
#print(upp_A)
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import ntpath
import os
import pandas as pd
import math
from statistics import mean

matrix = np.zeros([100,100])
#four_vecs_per_subject = np.zeros([4950,4])
#k=0
#path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all"
#for filename in glob.glob(os.path.join(path, '101309*17Networks_order_FSLMNI152_2mm_BOLD.csv')): 
data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all\101309_rfMRI_REST1_RL_Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm_BOLD.csv", header = None)
for i in range(100):
    for j in range(100):
        vec1 = data.iloc[:,i].values.tolist()
        vec2 = data.iloc[:,j].values.tolist()
        matrix[i,j] = stats.pearsonr(np.array(vec1),np.array(vec2))[0]
vector = matrix[np.triu_indices(100)]
#four_vecs_per_subject[:,k] = vector #np.arctanh(vector)
#k+=1
print(matrix)
print(vector)
print(vector.shape)
#print(four_vecs_per_subject)
#avg_four = np.mean(four_vecs_per_subject, axis = 1)
#inv_tran_avg_four = np.tanh(avg_four)
#print(inv_tran_avg_four)

