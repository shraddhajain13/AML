
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


path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\zscored_BOLD" 
sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
for i in range(len(sub_num_list_old)):
    filee =  glob.glob(os.path.join(path, 'sub' + str((int)(sub_num_list_old[i])) + '*.csv'))
    #print(filee)
    with open(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Concatenated\conc' + str((int)(sub_num_list_old[i])) + '.csv'  , 'w') as outfile:
            for fname in filee:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)