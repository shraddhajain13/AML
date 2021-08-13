import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

atlas = ['S100', 'S200', 'S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
end_name_for_sc = ['SC_all', 'SC_all_Sch200', 'SC_all_Sch400', 'SC_all_Sch600', 'SC_all_HO', 'SC_HO_25th', 'SC_HO_35th', 'SC_HO_45th', 'SC_all_Shen79', 'SC_all_Shen156', 'SC_all_Shen232']
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
path_fc = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\FC_matrices_all_atlas\efc_"
path_sc = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_PL_matrices_all_atlas\esc_"
path_pl = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_PL_matrices_all_atlas\epl_"

for i in range(len(atlas)):
    print(atlas[i])
    all_corr_array = np.zeros([272, 4])
    data_efc = pd.read_csv(path_fc + atlas[i] + '.csv', header = None).values #has the efc (upper triangular part) for all subjects column wise
    data_esc = pd.read_csv(path_sc + end_name_for_sc[i] + '.csv', header = None).values #has the esc (upper triangular part) for all subjects column wise
    data_epl = pd.read_csv(path_pl + end_name_for_sc[i] + '.csv', header = None).values #has the epl (upper triangular part) for all subjects column wise

    corr_efc_esc_list = []
    corr_efc_epl_list = []
    corr_esc_epl_list = []
    
    for j in range(len(sub_num_list_old)):
        corr_efc_esc_list.append(stats.pearsonr(np.arctanh(data_efc[:, j]), data_esc[:, j])[0]) #calculating corr(eFC, eSC) after doing fisher Z for eFC because eFC is correlation too
        corr_efc_epl_list.append(stats.pearsonr(np.arctanh(data_efc[:, j]), data_epl[:, j])[0])
        corr_esc_epl_list.append(stats.pearsonr(data_esc[:, j], data_epl[:, j])[0])
        
    all_corr_array[:, 0] = sub_num_list_old
    all_corr_array[:, 1] = corr_efc_esc_list
    all_corr_array[:, 2] = corr_efc_epl_list
    all_corr_array[:, 3] = corr_esc_epl_list

    np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + '.csv', all_corr_array, delimiter = ',')





