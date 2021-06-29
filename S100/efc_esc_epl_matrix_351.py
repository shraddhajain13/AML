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
from maxcorr_gender_351_subs import sub_num_list_351_ordered


path1 = r"C:\Users\shrad\OneDrive\Desktop\Kyesam Data\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M"
path2 = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\HCP_365Subj_SC_Schaefer_10M\HCP_365Subj_SC_Schaefer_10M"

sub = 0
final_vec_fc = np.zeros([4950, 351])
final_vec_sc = np.zeros([4950, 351])
final_vec_pl = np.zeros([4950, 351])

for sub in range(len(sub_num_list_351_ordered)):
    subject_number = (int)(sub_num_list_351_ordered[sub])
    print(sub)
    file_fc = glob.glob(os.path.join(path1, str(subject_number)+ '*empFC_all'))
    file_sc = glob.glob(os.path.join(path2, str(subject_number)+ '*count.csv'))
    file_pl = glob.glob(os.path.join(path2, str(subject_number)+ '*length.csv'))
    data_fc = np.loadtxt(file_fc[0])
    data_sc = pd.read_csv(file_sc[0], header = None, delimiter = ' ')
    data_pl = pd.read_csv(file_pl[0], header = None, delimiter = ' ')
    matrix_sc = data_sc.values
    matrix_pl = data_pl.values 

    final_vec_fc[:,sub] = data_fc[np.triu_indices(100, 1)]
    final_vec_sc[:,sub] = matrix_sc[np.triu_indices(100, 1)]
    final_vec_pl[:,sub] = matrix_pl[np.triu_indices(100, 1)]
    sub = sub + 1

        
print(final_vec_fc)
#print(final_vec_sc)
#print(final_vec_pl)
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_concatenated_N351.csv',final_vec_fc,delimiter=",")    
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_N351.csv',final_vec_sc,delimiter=",")  
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_N351.csv',final_vec_pl,delimiter=",")
