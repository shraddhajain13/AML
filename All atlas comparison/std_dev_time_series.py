import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


path = r"D:\Shraddha\Concatenated_all_atlas"
atlas = ['S100', 'S200', 'S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
sub_num_list_old = np.loadtxt(r"D:\Data\List_23_28_54_49_118.txt", usecols=(0))

for i in range(len(atlas)):
    print(atlas[i])
    mean_std_dev = []
    median_std_dev = []
    for j in range(len(sub_num_list_old)):
        print(sub_num_list_old[j])
        std_dev_parcels = []
        path = os.path.join(r"D:\Shraddha\Concatenated_all_atlas", atlas[i])
        filename = os.path.join(path, 'conc_'+ str((int)(sub_num_list_old[j])) + '.csv')
        time_series_data = pd.read_csv(filename, header = None).values
        num_of_parcels = time_series_data.shape[1] 
        for k in range(num_of_parcels):
            std_dev_parcels.append(np.std(time_series_data[:, k])) #takes the std_dev of each time series
        mean_std_dev.append(np.mean(std_dev_parcels)) #stores the mean of the std_dev of all the time series, so one number per subject. Total length = 272
        median_std_dev.append(np.median(std_dev_parcels)) #stores the median of the std_dev of all the time series, so one number per subject. Total length = 272

    d1 = {'Subject' : np.array(sub_num_list_old), 'mean std dev of time series': np.array(mean_std_dev)}
    df1 = pd.DataFrame(data = d1) 
    df1.to_csv(r"D:\Shraddha\std_dev_all_atlas\mean_std_time_series_" + atlas[i] + '.csv', index = False)

    d2 = {'Subject' : np.array(sub_num_list_old), 'median std dev of time series' : np.array(median_std_dev)}
    df2 = pd.DataFrame(data = d2)   
    df2.to_csv(r"D:\Shraddha\std_dev_all_atlas\median_std_time_series_" + atlas[i] + '.csv', index = False)



r"""
file = r"D:\Shraddha\Concatenated_all_atlas\S200\conc_101309.csv"
data = pd.read_csv(file, header = None).values
std_dev = []
for i in range(200):
    arr = data[:, i]
    #print(arr)
    #std_dev.append(np.std(arr))
    print(np.std(arr))
    #break

#print(std_dev)       

"""   