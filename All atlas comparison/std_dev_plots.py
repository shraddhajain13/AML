import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath
import pingouin as pg
from statsmodels.stats import multitest


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

def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

std_eFC_male_all_atlas = np.zeros([128, 11])
std_eFC_female_all_atlas = np.zeros([144, 11])

std_time_series_male_all_atlas = np.zeros([128, 11])
std_time_series_female_all_atlas = np.zeros([144, 11])

eff_size_std_eFC = []
p_val_std_eFC = []

eff_size_std_time_series = []
p_val_std_time_series = []

corr_sfc_efc_phase = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values
corr_sfc_efc_lc = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values


for i in range(len(atlas)):
    std_eFC = pd.read_csv(os.path.join(path, "std_eFC_"+ atlas[i]) + '.csv').values[:, 1]
    mean_std_time_series = pd.read_csv(os.path.join(path, "mean_std_time_series_"+ atlas[i]) + '.csv').values[:, 1]
    corr_efc_esc = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + '.csv', header = None).values[:, 1]
    #print(corr_efc_esc)
    print(std_eFC)

    pearson_corr = stats.pearsonr(corr_sfc_efc_lc[:, i], corr_efc_esc)[0]
    
    std_eFC_male, std_eFC_female = categorise_male_female(std_eFC)
    std_eFC_male_all_atlas[:, i] = np.array(std_eFC_male)
    std_eFC_female_all_atlas[:, i] = np.array(std_eFC_female)
    t_value1, p_value1 = scipy.stats.ranksums(std_eFC_male, std_eFC_female)
    eff_size_std_eFC.append(pg.compute_effsize(std_eFC_male, std_eFC_female, eftype = 'hedges')) #effect size for coupling strength
    p_val_std_eFC.append(p_value1)

    mean_std_time_series_male, mean_std_time_series_female = categorise_male_female(mean_std_time_series)
    std_time_series_male_all_atlas[:, i] = np.array(mean_std_time_series_male)
    std_time_series_female_all_atlas[:, i] = np.array(mean_std_time_series_female)
    t_value2, p_value2 = scipy.stats.ranksums(mean_std_time_series_male, mean_std_time_series_female)
    eff_size_std_time_series.append(pg.compute_effsize(mean_std_time_series_male, mean_std_time_series_female, eftype = 'hedges')) #effect size for coupling strength
    p_val_std_time_series.append(p_value2)

    corr_sfc_efc_phase_male, corr_sfc_efc_phase_female = categorise_male_female(corr_sfc_efc_phase[:, i])
    corr_sfc_efc_lc_male, corr_sfc_efc_lc_female = categorise_male_female(corr_sfc_efc_lc[:, i])

    corr_efc_esc_male, corr_efc_esc_female = categorise_male_female(corr_efc_esc)

    #np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\male_data_std_efc_br.csv", std_eFC_male_all_atlas, delimiter = ',') 
    #np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\female_data_std_efc_br.csv", std_eFC_female_all_atlas, delimiter = ',')
    
    
    plt.rcParams['font.size'] = '20'
    plt.figure(figsize = (16, 8))
    plt.text(max(corr_efc_esc)*0.9, max(corr_sfc_efc_lc[:, i])*0.99, 'Pearsons corr = ' + str(round(pearson_corr, 2)), size = 20)#, ha = 'center', va = 'center')
    plt.plot(corr_efc_esc_male, corr_sfc_efc_lc_male, '.', label = 'Male')
    plt.plot(corr_efc_esc_female, corr_sfc_efc_lc_female, '.', label = 'Female')
    plt.title(atlas[i] + ' - Limit Cycle Model')
    plt.xlabel('corr(eFC, eSC)', fontsize = 20)
    plt.ylabel('corr(sFC, eFC)', fontsize = 20)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig(r"D:\Shraddha\Plots\Complexity\corr_sfc_efc_lc_vs_corr_efc_esc_" + atlas[i] + '.png')
    plt.show()
    
    

r"""
l = len(atlas)
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))


male_plots = plt.boxplot(std_time_series_male_all_atlas, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(std_time_series_female_all_atlas, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] #only for the purpose of x axis

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 45)
plt.xlim(-2, (l*2))
#plt.title(model + " - Before Regression")
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('Mean std dev of time series', fontsize = 20)
plt.tight_layout()
plt.show()
"""
r"""
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 

p_val_std_eFC = multitest.fdrcorrection(np.array(p_val_std_eFC))[1]
p_val_std_time_series = multitest.fdrcorrection(np.array(p_val_std_time_series))[1]
plt.rcParams['font.size'] = '20'
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (16, 9))
ax1.plot(np.array(atlas), np.array(eff_size_std_time_series), marker = '.', markersize = 10)
ax1.yaxis.grid()
ax1.set_title('Hedges g vs Atlas')
ax1.set_ylabel('Hedges g', fontsize = 20)

ax2.plot(np.array(atlas), np.array(p_val_std_time_series), marker = '.', markersize = 10)
ax2.set_title('P value (FDR corrected) vs Atlas')
ax2.set_ylabel('P value', fontsize = 20)
ax2.set_xlabel('Atlas', fontsize = 20)
ax2.axhline(y = 0.05, color = 'r', linestyle = '--')#, label = 'threshold $/alpha$')
ax2.yaxis.grid()
plt.xticks(rotation = 45)
#plt.legend()
#ax1.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

plt.tight_layout()
plt.show()
"""