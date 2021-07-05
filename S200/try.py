import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

data = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase\A9F02Ext100Tri_CPU100Tri_CPU2100Tri_CPU3100Tri_CPU4100Tri_CPU5_bif_max", usecols = (29, 30, 31, 32, 33, 34))
#print(data[0, 0])
#print(data[0, 1])
#print(data[0, 2])
k = 0
delay_fc_list = []
delay_sc_list = []
coup_str_fc_list = []
coup_str_sc_list = []
corr_sfc_efc_list = []
corr_sfc_esc_list = []

for i in range(272): #loop to extract the goodness of fit for each subject. It is the first row for each subject in the file of each parcellation.
    delay_sfc_efc = data[k, 0]
    coup_str_sfc_efc = data[k, 1]
    corr_sfc_efc = data[k, 2]
    delay_sfc_esc = data[k, 3]
    coup_str_sfc_esc = data[k, 4]
    corr_sfc_esc = data[k, 5]

    delay_fc_list.append(delay_sfc_efc)
    delay_sc_list.append(delay_sfc_esc)
    coup_str_fc_list.append(coup_str_sfc_efc)
    coup_str_sc_list.append(coup_str_sfc_esc)
    corr_sfc_efc_list.append(corr_sfc_efc)
    corr_sfc_esc_list.append(corr_sfc_esc)

    k = k + 25

print(corr_sfc_efc_list)
    