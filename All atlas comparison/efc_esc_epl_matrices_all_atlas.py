import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

path_list_for_fc = np.loadtxt("C:\\Users\\shrad\\Desktop\\Juelich\\Data\\path_fc.txt", dtype = str)
path_list_for_sc = np.loadtxt("C:\\Users\\shrad\\Desktop\\Juelich\\Data\\path_sc.txt", dtype = str)
sub_num_list_old = np.loadtxt("C:\\Users\\shrad\\Desktop\\Juelich\\Data\\List_23_28_54_49_118.txt", usecols=(0))



#for path in path_list_for_fc: #has the path for the concatenated files obtained after zscoring
path = path_list_for_fc[1]
atlas = str(ntpath.basename(path))
print(atlas) 
efc_matrix = []
for sub in sub_num_list_old:
    sub = str((int)(sub))
    print(sub)
    #for filename in glob.glob(os.path.join(path, '*'+ sub + '*.csv'):
    file_name = os.path.join(path, 'conc_'+ sub + '.csv')
    data_fc = pd.read_csv(file_name, header = None).values
    matrix_fc = np.zeros([data_fc.shape[1], data_fc.shape[1]])
    #print(data_fc.shape[1])
    for i in range(data_fc.shape[1]):
        for j in range(data_fc.shape[1]):
            matrix_fc[i][j] = stats.spearmanr(data_fc[:, i], data_fc[:, j])[0]

    upp_tri = matrix_fc[np.triu_indices(data_fc.shape[1], 1)]
    #print(len(upp_tri))
    efc_matrix.append(upp_tri)
np.savetxt(r"D:\Shraddha\FC_matrices_all_atlas_spearman\efc_sp_" + atlas + ".csv", np.array(efc_matrix).transpose(), delimiter = ',')


r"""
i = 0
def sc_matrix():
    for path in path_list_for_sc:
        atlas = str(ntpath.basename(path))
        print(atlas, i)
        esc_matrix = []
        epl_matrix = []
        for sub in sub_num_list_old:
            sub = str((int)(sub))
            print(sub)

            if(i < 4):
                file_sc = glob.glob(os.path.join(path, sub + '*17Networks_order_FSLMNI152_1mm_count.csv'))[0]
                file_pl = glob.glob(os.path.join(path, sub + '*17Networks_order_FSLMNI152_1mm_length.csv'))[0]
            else:
                file_sc = glob.glob(os.path.join(path, sub + '*count.csv'))[0]
                file_pl = glob.glob(os.path.join(path, sub + '*length.csv'))[0]
            #print(file_sc)
            #print(file_pl)
            data_sc = pd.read_csv(file_sc, header = None, delimiter = ' ')
            data_pl = pd.read_csv(file_pl, header = None, delimiter = ' ')
            matrix_sc = data_sc.values
            matrix_pl = data_pl.values
            esc_matrix.append(matrix_sc[np.triu_indices(data_sc.shape[1], 1)])
            epl_matrix.append(matrix_pl[np.triu_indices(data_pl.shape[1], 1)])
        i = i + 1
        np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_PL_matrices_all_atlas\esc_" + atlas + ".csv", np.array(esc_matrix).transpose(), delimiter = ',')   
        np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\SC_PL_matrices_all_atlas\epl_" + atlas + ".csv", np.array(epl_matrix).transpose(), delimiter = ',')
"""