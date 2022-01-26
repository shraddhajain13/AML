import numpy as np
import pandas as pd
import os

data = pd.read_csv(r"D:\Shraddha\FC_matrices_all_atlas\efc_S200.csv", header = None).values
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))
threshold = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

#print(data.shape)
for k in range(len(threshold)):

    storing_efc = np.zeros([data.shape[0], data.shape[1]])
    for i in range(len(sub_num_list_old)):
        efc = data[:, i]
        efc[abs(efc) < threshold[k]] = 0
        storing_efc[:, i] = efc 
    #print(storing_efc.shape)
    print(threshold[k])
    np.savetxt("D:\\Shraddha\\Thresholding\\thresholded_efc_" + str(threshold[k]) + ".csv", storing_efc, delimiter = ',')
    

