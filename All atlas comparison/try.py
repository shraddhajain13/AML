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

