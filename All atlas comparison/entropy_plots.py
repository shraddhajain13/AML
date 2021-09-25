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
import statistics
import pickle 
import plotly 
import pingouin as pg
from statsmodels.stats import multitest


atlas = ['S100', 'S200', 'S400', 'S600', 'HO0', 'HO25', 'HO35', 'HO45', 'Shen79', 'Shen156', 'Shen232']
path_shan = r"D:\Shraddha\shan_entr_all_atlas\shan_entr_eFC_"
path_sampent = r"D:\Shraddha\sample_entropy_all_atlas\mean_sample_entropy_time_series_"

#data = pd.read_csv(r"D:\Shraddha\sample_entropy_all_atlas\mean_sample_entropy_time_series_" + atlas[0] + '.csv').values[:, 1]
#print(data)

sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]

corr_sfc_efc_phase = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values
corr_sfc_efc_lc = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values

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

shan_ent_male_all_atlas = np.zeros([128, 11])
shan_ent_female_all_atlas = np.zeros([144, 11])

samp_ent_male_all_atlas = np.zeros([128, 11])
samp_ent_female_all_atlas = np.zeros([144, 11])

def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

eff_size_shan_ent = []
eff_size_samp_ent = []
p_val_shan = []
p_val_samp = []

for i in range(len(atlas)):
    print(atlas[i])
    shan_ent = pd.read_csv(path_shan + atlas[i] + '.csv').values[:, 1]
    samp_ent = pd.read_csv(path_sampent + atlas[i] + '.csv').values[:, 1]
    corr_efc_esc = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + '.csv', header = None).values[:, 1]


    shan_ent_male, shan_ent_female = categorise_male_female(shan_ent)
    samp_ent_male, samp_ent_female = categorise_male_female(samp_ent)

    corr_sfc_efc_phase_male, corr_sfc_efc_phase_female = categorise_male_female(corr_sfc_efc_phase[:, i])
    corr_sfc_efc_lc_male, corr_sfc_efc_lc_female = categorise_male_female(corr_sfc_efc_lc[:, i])

    corr_efc_esc_male, corr_efc_esc_female = categorise_male_female(corr_efc_esc)

    #corr = stats.pearsonr(shan_ent, corr_efc_esc)[0]

    t_value_shan, p_value_shan = scipy.stats.ranksums(shan_ent_male, shan_ent_female)
    eff_size_shan_ent.append(pg.compute_effsize(shan_ent_male, shan_ent_female, eftype = 'hedges'))
    p_val_shan.append(p_value_shan)

    t_value_samp, p_value_samp = scipy.stats.ranksums(samp_ent_male, samp_ent_female)
    eff_size_samp_ent.append(pg.compute_effsize(samp_ent_male, samp_ent_female, eftype = 'hedges'))
    p_val_samp.append(p_value_samp)

    #shan_ent_male_all_atlas[:, i] = np.array(shan_ent_male)
    #shan_ent_female_all_atlas[:, i] = np.array(shan_ent_female)

    #samp_ent_male_all_atlas[:, i] = np.array(samp_ent_male)
    #samp_ent_female_all_atlas[:, i] = np.array(samp_ent_female)

    #plt.rcParams['font.size'] = '20'
    #plt.figure(figsize = (16, 9))
    #plt.text(max(corr_efc_esc)*0.92, max(shan_ent)*0.99, 'Pearsons corr = ' + str(round(corr, 2)), size = 20)
    #plt.plot(corr_efc_esc_male, shan_ent_male, '.', label = 'Male')
    #plt.plot(corr_efc_esc_female, shan_ent_female, '.', label = 'Female')
    #plt.title(atlas[i])
    #plt.xlabel('corr(eFC, eSC)', fontsize = 20)
    #plt.ylabel('Shannon entropy of eFC', fontsize = 20)
    #plt.legend(loc = 'upper left')
    #plt.tight_layout()
    #plt.savefig(r"D:\Shraddha\Plots\Complexity\entropy\shan_ent_vs_corr_efc_esc_" + atlas[i] + '.png')
    #plt.show()

r"""
l = len(atlas)
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 9))


male_plots = plt.boxplot(samp_ent_male_all_atlas, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(samp_ent_female_all_atlas, positions = np.array(range(l))*2 + 0.3)

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
plt.ylabel('Sample entropy of BOLD time series', fontsize = 20)
plt.tight_layout()
plt.show()
"""

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 
#print(p_val_samp)
p_val_shan = multitest.fdrcorrection(np.array(p_val_shan))[1]
p_val_samp = multitest.fdrcorrection(np.array(p_val_samp))[1]
#print(p_val_samp)
plt.rcParams['font.size'] = '20'
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (16, 9))
ax1.plot(np.array(atlas), np.array(eff_size_samp_ent), marker = '.', markersize = 10)
ax1.yaxis.grid()
ax1.set_title('Hedges g vs Atlas - Sample entropy of BOLD time series')
ax1.set_ylabel('Hedges g', fontsize = 20)

ax2.plot(np.array(atlas), np.array(p_val_samp), marker = '.', markersize = 10)
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
