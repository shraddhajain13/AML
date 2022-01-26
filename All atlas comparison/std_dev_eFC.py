import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232','HO0', 'HO25', 'HO35', 'HO45']
#threshold = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

for i in range(len(atlas)):
    std_dev_list = []
    print(atlas[i])
    data_efc = pd.read_csv("D:\\Shraddha\\FC_matrices_all_atlas\\efc_" + atlas[i] + ".csv", header = None).values
    for j in range(len(sub_num_list_old)):
        print(sub_num_list_old[j])
        transformed_data = np.arctanh(data_efc[:, j]) #fisher z transform
        std_dev_list.append(np.tanh(np.std(abs(transformed_data)))) #first calculating the std and then back transform
    
    d = {'Subject': np.array(sub_num_list_old), 'std dev': np.array(std_dev_list)}
    df = pd.DataFrame(data = d)
    df.to_csv("D:\\Shraddha\\std_dev_all_atlas\\std_eFC_" + atlas[i] + '.csv', index = False)
    

