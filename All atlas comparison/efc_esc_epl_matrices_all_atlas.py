import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

#path_list_for_fc = np.loadtxt("C:\\Users\\shrad\\Desktop\\Juelich\\Data\\path_fc.txt", dtype = str)
path_list_for_sc = np.loadtxt("D:\\Shraddha\\Data\\path_sc.txt", dtype = str)
sub_num_list_old = np.loadtxt("D:\\Shraddha\\Data\\List_23_28_54_49_118.txt", usecols=(0))

r"""

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
    data_fc = pd.read_csv(file_name, header = None).values #has the time series for all parcellations within one atlas
    matrix_fc = np.zeros([data_fc.shape[1], data_fc.shape[1]]) #empty matrix for eFC calculation
    #print(data_fc.shape[1])
    for i in range(data_fc.shape[1]):
        for j in range(data_fc.shape[1]):
            matrix_fc[i][j] = stats.spearmanr(data_fc[:, i], data_fc[:, j])[0]

    upp_tri = matrix_fc[np.triu_indices(data_fc.shape[1], 1)] #extracting the upper triangular part of the eFC matrix of size data_fc.shape[1] and the second argument being 1 implies exclusion of the diagonal
    #print(len(upp_tri))
    efc_matrix.append(upp_tri)
np.savetxt(r"D:\Shraddha\FC_matrices_all_atlas_spearman\efc_sp_" + atlas + ".csv", np.array(efc_matrix).transpose(), delimiter = ',')
"""

parcel = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45']

def sc_matrix(): #extraction of the eSC and ePL matrix
    for i, path in enumerate(path_list_for_sc):
        atlas = str(ntpath.basename(path)) #last part of the path 
        print(atlas, i)
        print(parcel[i])
        #esc_matrix = []
        #epl_matrix = []
        #for each atlas, we loop over all subjects
        for sub in sub_num_list_old:
            sub = str((int)(sub))
            print(sub)

            if(i < 4): #the files are named differently from Shen79 atlas onwards till HO45
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
            #print("eSC: ", matrix_sc.shape)
            #print("ePL: ", matrix_pl.shape)
            
            #esc_matrix.append(matrix_sc[np.triu_indices(data_sc.shape[1], 1)]) #extracting the upper triangular part (save it as transpose)
            #epl_matrix.append(matrix_pl[np.triu_indices(data_pl.shape[1], 1)]) #extracting the upper triangular part (save it as transpose)
        
            np.savetxt("D:\\Shraddha\\SC_PL_matrices_all_atlas\\full_matrices\\eSC_full\\" + parcel[i] + "\\esc_" + sub + ".csv", matrix_sc, delimiter = ',')   
            np.savetxt("D:\\Shraddha\\SC_PL_matrices_all_atlas\\full_matrices\\ePL_full\\" + parcel[i] + "\\epl_" + sub + ".csv", matrix_pl, delimiter = ',')   
sc_matrix()
