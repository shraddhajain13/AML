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


sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))  ## this is the list of 272 subjects that we want to investigate 
path1 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all"

for sub in range(len(sub_num_list_old)):
    subject_number = (int)(sub_num_list_old[sub])
    #print(subject_number)
    fiile = glob.glob(os.path.join(path1, str(subject_number)+ '*17Networks_order_FSLMNI152_2mm_BOLD.csv')) ## fiile contains the list of all files for one subject
    #print(fiile)
    k = 0
    for file_name in fiile: #going through each of the 4 files for one subject. this loop iterates 4 times for every subject
        data = pd.read_csv(file_name, header = None)
        zscored_data = scipy.stats.zscore(data.values)
        np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\zscored_BOLD\sub' + str(subject_number) + str(k) + '.csv', zscored_data, delimiter=",") 
        k = k+1
        