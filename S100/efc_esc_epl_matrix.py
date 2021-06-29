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
final_vec_pl = np.zeros([4950, 272])

for sub in range(len(sub_num_list_old)):
    subject_number = (int)(sub_num_list_old[sub])
    print(sub)
    file_sc = glob.glob(os.path.join(path2, str(subject_number)+ '*17Networks_order_FSLMNI152_1mm_count.csv'))
    file_pl = glob.glob(os.path.join(path2, str(subject_number)+ '*17Networks_order_FSLMNI152_1mm_length.csv'))
    data_sc = pd.read_csv(file_sc[0], header = None, delimiter = ' ')
    data_pl = pd.read_csv(file_pl[0], header = None, delimiter = ' ')
    matrix_sc = data_sc.values
    matrix_pl = data_pl.values
    fiile = glob.glob(os.path.join(path1, str(subject_number)+ '*17Networks_order_FSLMNI152_2mm_BOLD.csv')) ## fiile contains the list of all files for one subject
    #print(fiile)
    four_vecs_per_subject = np.zeros([4950,4])
    k = 0
    for file_name in fiile: #going through each of the 4 files for one subject. this loop iterates 4 times for every subject
        matrix = np.zeros([100,100]) 
        data = pd.read_csv(file_name, header = None)
        for i in range(100): ##building matrix per file 
            for j in range(100):
                vec1 = data.iloc[:,i].values
                vec2 = data.iloc[:,j].values
                matrix[i][j] = stats.pearsonr(vec1,vec2)[0]
        upp_tri_vec_fc = matrix[np.triu_indices(100, 1)]
        four_vecs_per_subject[:,k] = np.arctanh(upp_tri_vec_fc) # collects the 4 vectors for one subject after fisher z transform
        k = k+1
    avg_four = np.mean(four_vecs_per_subject, axis = 1)
    inv_tran_avg_four = np.tanh(avg_four)
    final_vec_fc[:,sub] = inv_tran_avg_four
    final_vec_sc[:,sub] = matrix_sc[np.triu_indices(100, 1)]
    final_vec_pl[:,sub] = matrix_pl[np.triu_indices(100, 1)]
    sub = sub + 1

        
print(final_vec_fc)
#print(final_vec_sc)
#print(final_vec_pl)
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_with_fisher_z.csv',final_vec_fc,delimiter=",")    
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_retry.csv',final_vec_sc,delimiter=",")  
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_retry.csv',final_vec_pl,delimiter=",")
 