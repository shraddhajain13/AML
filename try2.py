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
path2 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_all\SC_all"
sub_num_list_new = []
sub = 0
final_vec_fc = np.zeros([4950,272])
final_vec_sc = np.zeros([4950,272])
for filename in glob.glob(os.path.join(path1, '*17Networks_order_FSLMNI152_2mm_BOLD.csv')): 
    subject_number = (int)(ntpath.basename(filename)[0:6])
    if subject_number in sub_num_list_new:
        continue
    sub_num_list_new.append(subject_number)
    if subject_number in sub_num_list_old: #taking only subjects of interest (total = 272)
        
        file_sc = glob.glob(os.path.join(path2, str(subject_number)+ '*17Networks_order_FSLMNI152_1mm_count.csv'))
        data_sc = pd.read_csv(file_sc[0], header = None, delimiter = ' ')
        matrix_sc = data_sc.values
        final_vec_sc[:,sub] = matrix_sc[np.triu_indices(100, 1)]
        print(matrix_sc[np.triu_indices(100, 1)])
        sub = sub+1
print(final_vec_sc)