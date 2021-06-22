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
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Simulated_BOLD\sim_bold_int_1point5"
for i in range(len(sub_num_list_old)):
    subject_number = str((int)(sub_num_list_old[i]))
    for filename in glob.glob(os.path.join(path, 'sim_bold_1point5_' + subject_number + '.csv')):
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
#print(vec_fc)
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_sim_int_1point5.csv', vec_fc, delimiter=",")   
r"""
from Entropy import synchroni
data1 = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_sim.csv", header = None).values
data2 = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_zscored.csv", header = None).values
corr_nfc_rfc = np.zeros(272)
mean_nfc = np.zeros(272)
for i in range(272):
    corr_nfc_rfc[i] =  stats.pearsonr(data1[:,i], data2[:,i])[0]
    mean_nfc[i] = np.mean(abs(data1[:,i]))
#print(mean_nfc)
print(stats.pearsonr(mean_nfc, synchroni)[0])
x = np.linspace(1,272,272)
plt.plot(x, corr_nfc_rfc, '.')
plt.xlabel('Subject number')
plt.ylabel('corr(noisy_fc, real_fc')
plt.show()
#print(min(data2[:,0]))
#print(min(data1[:,0]))
"""