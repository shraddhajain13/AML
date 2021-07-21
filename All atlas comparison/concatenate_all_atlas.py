import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


path_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\path_conc.txt", dtype = str) #path_conc has the path to the zscored BOLD series
#print(path_list)
saving_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\save_list.txt", dtype = str)
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))

i = 0

for path in path_list:
    #print(path)
    for sub in sub_num_list_old:
        filee =  glob.glob(os.path.join(path, '*' + str((int)(sub)) + '*.csv'))
        print(filee)
        with open(saving_list[i] + str((int)(sub)) + '.csv'  , 'w') as outfile:
                for fname in filee:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
    i = i + 1
    