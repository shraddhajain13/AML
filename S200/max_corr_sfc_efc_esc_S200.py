import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


pheno_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
sub_num_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))

path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase"

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

gender_list_filtered = rearr(gender_list) #rearranging the gender list from the phenotypical data into the order of the simulated subjects

def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
        list1 = [] #for males
        list2 = [] #for females
        for i in range(272):
            if(gender_list_filtered[i] == 'M'):
                list1.append(x[i])
            if(gender_list_filtered[i] == 'F'):
                list2.append(x[i])
        return list1, list2

for filename in glob.glob(os.path.join(path, '*bif_max')):
    #print(type(filename))
    fiile = str(ntpath.basename(filename))
    data = np.loadtxt(filename, usecols = (29, 30, 31, 32, 33, 34))
    k = 0
    delay_fc_list = []
    delay_sc_list = []
    coup_str_fc_list = []
    coup_str_sc_list = []
    corr_sfc_efc_list = []
    corr_sfc_esc_list = []

    for i in range(272): #loop to extract the goodness of fit for each subject. It is the first row for each subject in the file of each parcellation.
        delay_sfc_efc = data[k, 0]
        coup_str_sfc_efc = data[k, 1]
        corr_sfc_efc = data[k, 2]
        delay_sfc_esc = data[k, 3]
        coup_str_sfc_esc = data[k, 4]
        corr_sfc_esc = data[k, 5]

        delay_fc_list.append(delay_sfc_efc)
        delay_sc_list.append(delay_sfc_esc)
        coup_str_fc_list.append(coup_str_sfc_efc)
        coup_str_sc_list.append(coup_str_sfc_esc)
        corr_sfc_efc_list.append(corr_sfc_efc)
        corr_sfc_esc_list.append(corr_sfc_esc)

        k = k + 25

    #delay_male, delay_female = categorise_male_female(delay_list)
    #coup_str_male, coup_str_female = categorise_male_female(coup_str_list)
    corr_sfc_efc_male, corr_sfc_efc_female = categorise_male_female(corr_sfc_efc_list)
    corr_sfc_esc_male, corr_sfc_esc_female = categorise_male_female(corr_sfc_esc_list)

    t_value_fc, p_value_fc = scipy.stats.ttest_ind(corr_sfc_efc_male, corr_sfc_efc_female) #two tailed t test for corr(sFC, eFC)
    t_value_sc, p_value_sc = scipy.stats.ttest_ind(corr_sfc_esc_male, corr_sfc_esc_female) #two tailed t test for corr(sFC, eSC)

    print(fiile)
    print('corr(sFC, eFC): ', t_value_fc, p_value_fc)
    print('corr(sFC, eSC): ', t_value_sc, p_value_sc)

    fc_data = [corr_sfc_efc_male, corr_sfc_efc_female] #for box plots
    sc_data = [corr_sfc_esc_male, corr_sfc_esc_female]

    fig, ax = plt.subplots(nrows = 1, ncols = 2, constrained_layout = True)
    #fig.tight_layout()
    fig.suptitle(fiile)
    ax[0].boxplot(fc_data)
    ax[0].set_xticklabels(['Male','Female'])
    ax[0].set_ylabel('Best fit correlation between sFC and eFC')
    ax[0].set_title('corr(sFC, eFC)')
    ax[1].boxplot(sc_data)
    ax[1].set_xticklabels(['Male','Female'])
    ax[1].set_ylabel('Best fit correlation between sFC and eSC')
    ax[1].set_title('corr(sFC, eSC)')
    #plt.savefig(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Plots\Box plots\box_plot" + fiile + ".png")
    #plt.show()
    
    
