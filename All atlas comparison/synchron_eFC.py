import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


path = r"D:\Shraddha\FC_matrices_all_atlas\efc_"
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232','HO0', 'HO25', 'HO35', 'HO45']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

for i in range(len(atlas)):
    print(atlas[i])
    synch_list = []
    data_efc = pd.read_csv(path + atlas[i] + '.csv', header = None).values
    for j in range(len(sub_num_list_old)):
        print(sub_num_list_old[j])
        #print(data_efc[:, j])
        synch_list.append(np.mean(abs(data_efc[:, j])))
    
    d = {'Subject': np.array(sub_num_list_old), 'synchronicity': np.array(synch_list)}
    df = pd.DataFrame(data = d)
    df.to_csv(r"D:\Shraddha\synch_all_atlas\synch_eFC_" + atlas[i] + '.csv', index = False)
    

