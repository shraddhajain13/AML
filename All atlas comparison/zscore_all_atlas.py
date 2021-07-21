import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath
import pingouin as pg

path_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\path_conc.txt", dtype = str)
#print(path_list)
saving_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\save_list_zscored.txt", dtype = str)
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))

i = 0

for path in path_list:
    for sub in sub_num_list_old:
        k = 1
        filee =  glob.glob(os.path.join(path, str((int)(sub)) + '*'))
        print(filee)
        for filename in filee:
            data = pd.read_csv(filename, header = None)
            zscored_data = scipy.stats.zscore(data.values)
            np.savetxt(saving_list[i] + str(sub) + '_' + str(k) + '.csv', zscored_data, delimiter=",") 
            k = k + 1
    i = i + 1