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

vectors_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_concatenated_N351.csv", header = None).values
vectors_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_N351.csv", header = None).values
vectors_pl = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_N351.csv", header = None).values

efc_esc = stats.pearsonr(vectors_fc[:,1], vectors_sc[:,1])[0]
efc_epl = stats.pearsonr(vectors_fc[:,1], vectors_pl[:,1])[0]
esc_epl = stats.pearsonr(vectors_sc[:,1], vectors_pl[:,1])[0]
print(efc_esc)
print(efc_epl)
print(esc_epl)


#path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_all\SC_all"
#i = 0
#for filename in glob.glob(os.path.join(path, '*17Networks_order_FSLMNI152_1mm_count.csv')):
    #print(filename)
    #i = i+1
#print(i)
#data = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Kyesam Data\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\951457_LR12-RL12_17Sch100B2mm_A9F0002_Del0-423Coupl0-504N03-10M_CPU_Nrm1000_StepInteg04_empFC_all")
#print(data)
#print(data.shape)
#upp_tri_part = data[np.triu_indices(100, 1)]
#print(upp_tri_part)
#array = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\HCP_included_351Subjects.txt")
#print(len(array))
#arr = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Kyesam Data\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\951457_LR12-RL12_17Sch100B2mm_A9F0002_Del0-423Coupl0-504N03-10M_CPU_Nrm1000_StepInteg04_bif_all",usecols=(0, 1, 2, 3))
#corr_sfc_efc = arr[:,2]
#corr_sfc_esc = arr[:,3]

#max_fc = max(corr_sfc_efc)
#max_sc = max(corr_sfc_esc)
#print(max_sc)
#max_corr_list_fc_351.append(max_fc)
#max_corr_list_sc_351.append(max_sc)