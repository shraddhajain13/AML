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
fisher_z_avg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_with_fisher_z.csv", header = None).values
simple_avg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_without_fisher_z.csv", header = None).values
zscored = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_zscored.csv", header = None).values
corr_12_list = []
corr_13_list = []
corr_23_list = []
diff_12_list = []
diff_13_list = []
diff_23_list = []
for i in range(272):
    corr12 = stats.pearsonr(fisher_z_avg[:,i], simple_avg[:,i])[0]
    corr13 = stats.pearsonr(fisher_z_avg[:,i], zscored[:,i])[0]
    corr23 = stats.pearsonr(simple_avg[:,i], zscored[:,i])[0]
    corr_12_list.append(corr12)
    corr_13_list.append(corr13)
    corr_23_list.append(corr23)
    diff12 = fisher_z_avg[:,i] - simple_avg[:,i]
    diff13 = fisher_z_avg[:,i] - zscored[:,i]
    diff23 = simple_avg[:,i] - zscored[:,i]
    diff_12_list.append(diff12)
    diff_13_list.append(diff13)
    diff_23_list.append(diff23)
plt.plot(sub_num_list_old, corr_23_list, '.')
plt.xlabel('Subject number')
plt.ylabel('correlation coefficient')
plt.title('Correlation between eFCs without fisher z transform (involves averaging over 4 sessions) and z scoring (concatenation)')
plt.show()