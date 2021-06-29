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
#from multi_linear_reg import sub_num_list_old

sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all"
sub_num_list_new = []
for filename in glob.glob(os.path.join(path, '*17Networks_order_FSLMNI152_2mm_BOLD.csv')): 
    subject_number = (int)(ntpath.basename(filename)[0:6])
    if subject_number in sub_num_list_new:
        continue 
    sub_num_list_new.append(subject_number)
    if subject_number in sub_num_list_old:
        filee = glob.glob(os.path.join(path, str(subject_number)+'*17Networks_order_FSLMNI152_2mm_BOLD.csv'))
        with open(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Concatenated' + str(subject_number) , 'w') as outfile:
            for fname in filee:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)