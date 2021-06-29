import numpy as np
import math
import pandas as pd


pheno_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
sub_num_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
data = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase\A9F0002Ext200Tri_GPU200Tri_GPU2200Tri_GPU3200Tri_GPU4200Tri_GPU5_bif_max", usecols = (0, 1, 2, 3, 4))
k = 0
delay_list = []
coup_str_list = []
corr_sfc_efc_list = []
corr_sfc_esc_list = []

def rearr(list_1):
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)
    

for i in range(272):
    delay = data[k, 1]
    coup_str = data[k, 2]
    corr_sfc_efc = data[k, 3]
    corr_sfc_esc = data[k, 4]

    delay_list.append(delay)
    coup_str_list.append(coup_str)
    corr_sfc_efc_list.append(corr_sfc_efc)
    corr_sfc_esc_list.append(corr_sfc_esc)

    k = k + 25

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

delay_male, delay_female = categorise_male_female(delay_list)
coup_str_male, coup_str_female = categorise_male_female(coup_str_list)
corr_sfc_efc_male, corr_sfc_efc_female = categorise_male_female(corr_sfc_efc_list)
corr_sfc_esc_male, corr_sfc_esc_female = categorise_male_female(corr_sfc_esc_list)
