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
 
sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
vec_fc = np.zeros([4950,272])
sub = 0
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Concatenated"
for i in range(len(sub_num_list_old)):
    subject_number = str((int)(sub_num_list_old[i]))
    for filename in glob.glob(os.path.join(path, 'conc' + subject_number + '.csv')):
        print(subject_number)
        matrix = np.zeros([100, 100])
        data = pd.read_csv(filename, header = None)
        for i in range(100): ##building matrix per file 
            for j in range(100):
                vec1 = data.iloc[:,i].values
                vec2 = data.iloc[:,j].values 
                matrix[i][j] = stats.pearsonr(vec1,vec2)[0]
        upp_tri = matrix[np.triu_indices(100, 1)]
        vec_fc[:,sub] = upp_tri
        sub = sub + 1
print(vec_fc)
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_zscored.csv', vec_fc, delimiter=",")   