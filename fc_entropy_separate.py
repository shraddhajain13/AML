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
import antropy as ant
from multi_linear_reg import gender_list_filtered, brain_size_list

r"""
x = np.linspace(-1,1,21)
def entr_fc_sc(vec):
    counts = plt.hist(vec, bins = x)[0]
    entr = scipy.stats.entropy(counts) #shannon entropy of FC/SC
    return entr


def modified_sc(sc_matrix):
    mat = np.zeros([100,100])
    for i in range(100):
        for j in range(100):
            mat[i][j] = np.log(sc_matrix[i][j]+1)
    return mat

sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))  ## this is the list of 272 subjects that we want to investigate 
path1 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\BOLD_time_all\BOLD_time_all"
path2 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_all\SC_all"
sub_num_list_new = []
sub = 0
#final_vec_fc = np.zeros([4950,272])
entropy_fc = []
entropy_sc = []
samp_entr = []
#final_vec_sc = np.zeros([4950,272])
#final_vec_pl = np.zeros([4950, 272])

for sub in range(len(sub_num_list_old)):
    subject_number = (int)(sub_num_list_old[sub])
    print(sub)
    file_sc = glob.glob(os.path.join(path2, str(subject_number)+ '*17Networks_order_FSLMNI152_1mm_count.csv'))
    data_sc = pd.read_csv(file_sc[0], header = None, delimiter = ' ')
    matrix_sc = modified_sc(data_sc.values)
    fiile = glob.glob(os.path.join(path1, str(subject_number)+ '*17Networks_order_FSLMNI152_2mm_BOLD.csv')) ## fiile contains the list of all files for one subject
    for file_name in fiile: #going through each of the 4 files for one subject. this loop iterates 4 times for every subject
        matrix = np.zeros([100,100]) 
        data = pd.read_csv(file_name, header = None).values
        se = 0
        for k in range(100):
            se = se + ant.sample_entropy(data[:,k])
        samp_entr.append(se/100)
        for i in range(100): ##building matrix per file 
            for j in range(100):
                vec1 = data[:,i]
                vec2 = data[:,j]
                matrix[i][j] = stats.pearsonr(vec1,vec2)[0]
        upp_tri_vec_fc = matrix[np.triu_indices(100, 1)]
        upp_tri_vec_sc = matrix_sc[np.triu_indices(100, 1)]
        entropy_fc.append(entr_fc_sc(upp_tri_vec_fc))
    entropy_sc.append(entr_fc_sc(upp_tri_vec_sc))
        

np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\separate_entropy_fc.csv', entropy_fc, delimiter=",")  
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\separate_sample_entropy.csv', samp_entr, delimiter=",")
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sc.csv', entropy_sc, delimiter=",")  
"""

samp_ent = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\separate_sample_entropy.csv", header = None).values
ent_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\separate_entropy_fc.csv", header = None).values
ent_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sc.csv", header = None).values

k = 0
samp_ent_male_list = []
ent_fc_male_list = []
samp_ent_female_list = []
ent_fc_female_list = []

for i in range(len(gender_list_filtered)): 
    for j in range(4):
        if(gender_list_filtered[i] == 'M'):
            samp_ent_male_list.append(samp_ent[k])
            ent_fc_male_list.append(ent_fc[k])
        if(gender_list_filtered[i] == 'F'):
            samp_ent_female_list.append(samp_ent[k])
            ent_fc_female_list.append(ent_fc[k])  
        k = k+1
        print(k)
plt.plot(samp_ent_female_list, ent_fc_female_list, '.', label = 'Female')
plt.plot(samp_ent_male_list, ent_fc_male_list, '.', label = 'Male')
plt.xlabel('Sample entropy of BOLD')
plt.ylabel('Shannon entropy of FC')
plt.legend()
plt.show()
