import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath



path = r"D:\Shraddha\std_dev_all_atlas"
atlas = ['S100', 'S200', 'S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]


def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list_old[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

gender_list_filtered = rearr(gender_list)

def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    list1 = [] #for males
    list2 = [] #for females 
    for i in range(272):
        if(gender_list_filtered[i] == 'M'):
            list1.append(x[i])
        if(gender_list_filtered[i] == 'F'):
            list2.append(x[i])
    return list1, list2


for i in range(len(atlas)):
    std_eFC = pd.read_csv(os.path.join(path, "std_eFC_"+ atlas[i]) + '.csv').values[:, 1]
    mean_std_time_series = pd.read_csv(os.path.join(path, "mean_std_time_series_"+ atlas[i]) + '.csv').values[:, 1]
    
    std_eFC_male, std_eFC_female = categorise_male_female(std_eFC)
    mean_std_time_series_male, mean_std_time_series_female = categorise_male_female(mean_std_time_series)
    
    plt.rcParams['font.size'] = '20'
    plt.figure(figsize = (16, 8))

    plt.plot(mean_std_time_series_male, std_eFC_male, '.', label = 'Male')
    plt.plot(mean_std_time_series_female, std_eFC_female, '.', label = 'Female')
    plt.title(atlas[i])
    plt.xlabel('Mean std dev of time series data', fontsize = 20)
    plt.ylabel('Std dev of eFC', fontsize = 20)
    plt.tight_layout()
    plt.savefig(r"D:\Shraddha\Plots\Complexity\std_efc_vs_mean_std_time_series_" + atlas[i] + '.png')
    plt.show()
    
