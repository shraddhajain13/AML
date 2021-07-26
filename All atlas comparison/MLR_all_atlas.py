import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

sub_num_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
pheno_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
brain_size_list_full = pheno_data[:, 192] #list of all subject's brain sizes in the phenotypical data

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

brain_size_list_ordered = rearr(brain_size_list_full)

corr_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas.csv").values
corr_sfc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas.csv").values
corr_efc_esc_all_atlas = pd.read_csv()