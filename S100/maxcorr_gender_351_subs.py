import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import ntpath
import os
import pandas as pd
import math
from statistics import mean
import pingouin
from maxcorr_gender import sub_num_list_phen, gender_list
#from maxcorr_gender import max_corr_list_fc


num_subs = 351

max_corr_list_fc_351 = []
max_corr_list_sc_351 = []

avg_corr_list_fc_351 = []
avg_corr_list_sc_351 = []


sub_num_list_351_ordered = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\HCP_included_351Subjects.txt",usecols=(0))
path = r'C:\Users\shrad\OneDrive\Desktop\Kyesam Data\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M\HCP_MM_N365_Del0-423_Coupl0-504_NmzSCmean_NmzPL1000_WBT10M'

for i in range(num_subs):
    sub_num = (int)(sub_num_list_351_ordered[i])
    #print(sub_num)
    for filename in glob.glob(os.path.join(path, str(sub_num) + '*Sch*bif_all')):
        arr = np.loadtxt(filename,usecols=(0, 1, 2, 3))
        corr_sfc_efc = arr[:,2]
        corr_sfc_esc = arr[:,3]
        
        max_fc = max(corr_sfc_efc)
        max_sc = max(corr_sfc_esc)
        
        avg_fc = mean(corr_sfc_efc)
        avg_sc = mean(corr_sfc_esc)

        max_corr_list_fc_351.append(max_fc)
        max_corr_list_sc_351.append(max_sc)

        avg_corr_list_fc_351.append(avg_fc)
        avg_corr_list_sc_351.append(avg_sc)

        
    





r"""
    for filename in glob.glob(os.path.join(path, '*Sch*bif_all')):
        sub_num = (int)(ntpath.basename(filename)[0:6])
        sub_num_list_351.append(sub_num)
        arr = np.loadtxt(filename,usecols=(0, 1, 2, 3))
        corr_sfc_efc = arr[:,2]
        corr_sfc_esc = arr[:,3]
        
        max_fc = max(corr_sfc_efc)
        max_sc = max(corr_sfc_esc)
        
        max_corr_list_fc_351.append(max_fc)
        max_corr_list_sc_351.append(max_sc)

    
def rearr(list_1):
    list_2 = []
    for i in range(351):
        ind = np.where(sub_num_list_351 == sub_num_list_351_ordered[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

corr_sfc_efc_list_365 = rearr(max_corr_list_fc_365) #ordered list
corr_sfc_esc_list_365 = rearr(max_corr_list_sc_365)

#print(len(corr_sfc_efc_list_365))
"""

def male_female_classify(l1):
    return_male = []
    return_female = []
    for i in range(351):
        index = sub_num_list_phen.index(sub_num_list_351_ordered[i])
        gen = gender_list[index]
        if(gen == 'M'):
            return_male.append(l1[i])
        if(gen == 'F'):
            return_female.append(l1[i])
    return return_male, return_female

male_sfc_efc_list, female_sfc_efc_list = male_female_classify(avg_corr_list_fc_351)
male_sfc_esc_list, female_sfc_esc_list = male_female_classify(avg_corr_list_sc_351)

t, p = scipy.stats.ttest_ind(male_sfc_esc_list, female_sfc_esc_list)
eff_size = pingouin.compute_effsize(male_sfc_esc_list, female_sfc_esc_list)
#print("t = ", t)
#print("p = ", p)
#print("Effect size = ", eff_size)

r"""
fc_data = [male_sfc_efc_list, female_sfc_efc_list]
sc_data = [male_sfc_esc_list, female_sfc_esc_list]
fig, ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].boxplot(fc_data)
ax[0].set_xticklabels(['Male','Female'])
ax[0].set_ylabel('Best fit correlation between sFC and eFC - 365 subjects')
ax[0].set_title('FC')
ax[1].boxplot(sc_data)
ax[1].set_xticklabels(['Male','Female'])
ax[1].set_ylabel('Best fit correlation between sFC and eSC')
ax[1].set_title('SC')
plt.show()
"""