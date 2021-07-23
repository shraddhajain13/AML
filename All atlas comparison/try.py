import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

r"""

path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\HCP_Deriv\BOLD_time_17N_all\BOLD_time_17N_all"
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))

for sub in sub_num_list_old:
    k = 1
    sub = str((int)(sub)) 
    print(sub)
    for filename in glob.glob(os.path.join(path, sub + '*')):
        data = pd.read_csv(filename, header = None)
        zscored_data = scipy.stats.zscore(data.values)
        np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\temp_zscore_S100\zscore_" + str(sub) + '_' + str(k) + '.csv', zscored_data, delimiter=",") 
        k = k + 1

y = []
for i in range(20):
    x = np.linspace(1, 10, 10).tolist()
    y.append(x)

np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\tryy.csv", np.array(y).transpose(), delimiter = ',')
#z = np.array([[1,2,3], [4,5,6]]).transpose()
#print(type(np.c_[z]))
#print(z)
"""
r"""
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
path = "C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\Concatenated_all_atlas\\HO25"
efc_matrix = []
for sub in sub_num_list_old:
    sub = str((int)(sub))
    print(sub)
    file_name = os.path.join(path, 'conc_'+ sub + '.csv')
    data_fc = pd.read_csv(file_name, header = None).values
    matrix_fc = np.zeros([data_fc.shape[1], data_fc.shape[1]])
    #print(data_fc.shape[1])
    for i in range(data_fc.shape[1]):
        for j in range(data_fc.shape[1]):
            matrix_fc[i][j] = stats.pearsonr(data_fc[:, i], data_fc[:, j])[0]

    upp_tri = matrix_fc[np.triu_indices(data_fc.shape[1], 1)]
    #print(len(upp_tri))
    efc_matrix.append(upp_tri)
np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\efc_HO25_temp.csv", np.array(efc_matrix).transpose(), delimiter = ',')
"""
for i in range(10):
    if i<4:
        print(i)