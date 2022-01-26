import numpy as np
import pandas as pd
import scipy
import glob
from scipy import stats
import matplotlib.pyplot as plt
import collections
import ntpath
import os
import pandas as pd
import math
import antropy as ant
#from multi_linear_reg import corr_sfc_efc_list, corr_sfc_esc_list, corr_eFC_eSC_list, corr_eFC_ePL_list, corr_eSC_ePL_list, gender_list_filtered, brain_size_list
import statistics
import pickle 
import plotly 
#from mpl_toolkits.mplot3d import Axes3D
#import plotly.express as px

r"""
x = np.linspace(-1, 1, 21) #patition into 20 bins from -1 to 1
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

threshold = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
for i in range(len(threshold)):
    shan_entr_efc = []
    data_efc = pd.read_csv("D:\\Shraddha\\Thresholding\\thresholded_efc\\thresholded_efc_" + str(threshold[i]) + ".csv", header = None).values
    #print(data_efc.shape)
    print(threshold[i])
    for j in range(len(sub_num_list_old)):
        print(sub_num_list_old[j])
        transformed_data = np.arctanh(data_efc[:, j]) #fisher z transform
        counts = plt.hist(transformed_data, bins = x)[0] #counting the number of points in each bin defined by x
        shan_entr_efc.append(scipy.stats.entropy(counts))

    d = {'Subject': np.array(sub_num_list_old), 'Shannon entropy of eFC': np.array(shan_entr_efc)}
    df = pd.DataFrame(data = d)
    df.to_csv("D:\\Shraddha\Thresholding\\shan_entropy\\shan_entr_" + str(threshold[i]) + ".csv", index = False)
"""

r"""
atls = ['SC_all', 'SC_all_Sch200', 'SC_all_Sch400', 'SC_all_Sch600', 'SC_all_HO', 'SC_HO_25th', 'SC_HO_35th', 'SC_HO_45th', 'SC_all_Shen79', 'SC_all_Shen156', 'SC_all_Shen232']
def shannon_entropy_sc():
    path = r"D:\Shraddha\SC_PL_matrices_all_atlas\esc_"
    for i in range(len(atls)):
        print(atls[i])
        shan_entr_esc = []
        data_esc = pd.read_csv(path + atls[i] + '.csv', header = None).values
        for j in range(len(sub_num_list_old)):
            print(sub_num_list_old[j])
            counts = plt.hist(data_esc[:, j], bins = x)[0] #counting the number of points in each bin defined by x
            shan_entr_esc.append(scipy.stats.entropy(counts))

        d = {'Subject': np.array(sub_num_list_old), 'Shannon entropy of eSC': np.array(shan_entr_esc)}
        df = pd.DataFrame(data = d)
        df.to_csv(r"D:\Shraddha\shan_entr_all_atlas\shan_entr_eSC_" + atlas[i] + '.csv', index = False)
"""
r"""
path = r"D:\Shraddha\Concatenated_all_atlas"
atlas = ['S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
sub_num_list_old = np.loadtxt(r"D:\Data\List_23_28_54_49_118.txt", usecols=(0))

for i in range(len(atlas)):
    print(atlas[i])
    mean_samp_ent = []
    median_samp_ent = []
    for j in range(len(sub_num_list_old)):
        print(sub_num_list_old[j])
        samp_ent_parcels = []
        path = os.path.join(r"D:\Shraddha\Concatenated_all_atlas", atlas[i])
        filename = os.path.join(path, 'conc_'+ str((int)(sub_num_list_old[j])) + '.csv')
        time_series_data = pd.read_csv(filename, header = None).values
        num_of_parcels = time_series_data.shape[1] 
        for k in range(num_of_parcels):
            samp_ent_parcels.append(ant.sample_entropy(time_series_data[:, k])) #takes the sample entropy of each time series
        mean_samp_ent.append(np.mean(samp_ent_parcels)) #stores the mean of the std_dev of all the time series, so one number per subject. Total length = 272
        median_samp_ent.append(np.median(samp_ent_parcels)) #stores the median of the std_dev of all the time series, so one number per subject. Total length = 272

    d1 = {'Subject' : np.array(sub_num_list_old), 'mean sample entropy of time series': np.array(mean_samp_ent)}
    df1 = pd.DataFrame(data = d1) 
    df1.to_csv(r"D:\Shraddha\sample_entropy_all_atlas\mean_sample_entropy_time_series_" + atlas[i] + '.csv', index = False)

    d2 = {'Subject' : np.array(sub_num_list_old), 'median sample entropy of time series' : np.array(median_samp_ent)}
    df2 = pd.DataFrame(data = d2)   
    df2.to_csv(r"D:\Shraddha\sample_entropy_all_atlas\median_sample_entropy_time_series_" + atlas[i] + '.csv', index = False)
"""
r"""
x = np.linspace(-1, 1, 21) #patition into 20 bins from -1 to 1
atlas = ['S100', 'S200', 'S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))


def shannon_entropy_fc():
    path = r"D:\Shraddha\FC_matrices_all_atlas\efc_"
    for i in range(len(atlas)):
        print(atlas[i])
        shan_entr_efc = []
        data_efc = pd.read_csv(path + atlas[i] + '.csv', header = None).values
        for j in range(len(sub_num_list_old)):

            print(sub_num_list_old[j])
            transformed_data = np.arctanh(data_efc[:, j]) #fisher z transform
            counts = plt.hist(transformed_data, bins = x)[0] #counting the number of points in each bin defined by x
            shan_entr_efc.append(scipy.stats.entropy(counts))

        d = {'Subject': np.array(sub_num_list_old), 'Shannon entropy of eFC': np.array(shan_entr_efc)}
        df = pd.DataFrame(data = d)
        df.to_csv(r"D:\Shraddha\shan_entr_all_atlas\shan_entr_eFC_" + atlas[i] + '.csv', index = False)


shannon_entropy_fc()
"""