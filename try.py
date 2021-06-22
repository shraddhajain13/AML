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
four_vecs_per_subject = np.zeros([4950,4])
k=0
path1 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all"
for filename in glob.glob(os.path.join(path1, '101309*17Networks_order_FSLMNI152_2mm_BOLD.csv')): 
    data = pd.read_csv(filename, header = None)
    for i in range(100):
        for j in range(100):
            vec1 = data.iloc[:,i].values
            vec2 = data.iloc[:,j].values
            matrix[i,j] = stats.pearsonr(vec1,vec2)[0]
    vector = matrix[np.triu_indices(100, 1)]
    four_vecs_per_subject[:,k] = np.arctanh(vector) #after fisher z transform
    k+=1

#print(four_vecs_per_subject)
avg_four = np.mean(four_vecs_per_subject, axis = 1)
inv_tran_avg_four = np.tanh(avg_four)
print(inv_tran_avg_four)
