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
from sklearn.linear_model import LinearRegression


path = r"D:\Shraddha\synch_all_atlas"
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232','HO0', 'HO25', 'HO35', 'HO45']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
brain_size_list_full = pheno_data[:, 192]

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list_old[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

gender_list_filtered = rearr(gender_list)
brain_size_list_sorted = rearr(brain_size_list_full)

def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    list1 = [] #for males
    list2 = [] #for females 
    for i in range(272):
        if(gender_list_filtered[i] == 'M'):
            list1.append(x[i])
        if(gender_list_filtered[i] == 'F'):
            list2.append(x[i])
    return list1, list2

def set_box_color(bp, color, flag): #setting color for box plots
    if flag == 1:
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    
    if flag == 0:
        plt.setp(bp['boxes'], color=color, linestyle = '--')
        plt.setp(bp['whiskers'], color=color, linestyle = '--')
        plt.setp(bp['caps'], color=color, linestyle = '--')
        plt.setp(bp['medians'], color=color, linestyle = '--')
    
    
synch_eFC_male_all_atlas = np.zeros([128, 11])
synch_eFC_female_all_atlas = np.zeros([144, 11])



eff_size_synch_eFC = []
p_val_synch_eFC = []


for i in range(len(atlas)):
    synch_eFC = pd.read_csv(os.path.join(path, "synch_eFC_"+ atlas[i]) + '.csv').values[:, 1]
    corr_efc_esc = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values[:, 1] #corr(eFC, eSC) for the given atlas
    
    
    ##########.....MLR....############
    
    X = np.zeros([272, 2])
    X[:, 0] = np.array(brain_size_list_sorted)
    X[:, 1] = np.array(np.arctanh(corr_efc_esc))

    Y = synch_eFC

    reg = LinearRegression().fit(X, Y)
    coef = reg.coef_
    Y_hat = np.dot(X, np.array(coef)) #retaining the costant in regression analysis

    residual = Y - Y_hat

    #resdiual = np.tanh(residual)

    #########.....end of MLR....#########
    
    
    synch_eFC_male, synch_eFC_female = categorise_male_female(residual)
    synch_eFC_male_all_atlas[:, i] = np.array(synch_eFC_male)
    synch_eFC_female_all_atlas[:, i] = np.array(synch_eFC_female)
    t_value1, p_value1 = scipy.stats.ranksums(synch_eFC_male, synch_eFC_female)
    eff_size_synch_eFC.append(pg.compute_effsize(synch_eFC_male, synch_eFC_female, eftype = 'hedges')) #effect size for coupling strength
    p_val_synch_eFC.append(p_value1)

    
    
    
    
    
#np.savetxt(r"D:\Shraddha\Data\male_data_synch_efc_br.csv", synch_eFC_male_all_atlas, delimiter = ',') #this storage is for the grey plots in the background
#np.savetxt(r"D:\Shraddha\Data\female_data_synch_efc_br.csv", synch_eFC_female_all_atlas, delimiter = ',') #this storage is for the grey plots in the background

l = len(atlas)
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))


male_data_br = pd.read_csv(r"D:\Shraddha\Data\male_data_synch_efc_br.csv", header = None).values #loading the ones before reg
female_data_br = pd.read_csv(r"D:\Shraddha\Data\female_data_synch_efc_br.csv", header = None).values #loading the ones before reg

male_plots_br = plt.boxplot(np.array(male_data_br), positions = np.array(range(l))*2 - 0.3, showfliers = False) #the ones before reg
female_plots_br = plt.boxplot(np.array(female_data_br), positions = np.array(range(l))*2 + 0.3, showfliers = False) #the ones before reg

male_plots = plt.boxplot(synch_eFC_male_all_atlas, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(synch_eFC_female_all_atlas, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue', 1) 
set_box_color(female_plots, 'red', 1)

set_box_color(male_plots_br, 'grey', 0) #before reg
set_box_color(female_plots_br, 'grey', 0) #before reg

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.plot([], '--', color = 'grey', label = 'Before regression')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] #only for the purpose of x axis

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('Synchronicity of eFC', fontsize = 20)
plt.tight_layout()
plt.show()

effsize_pval = np.zeros([l, 2])
effsize_pval[:, 0] = np.array(eff_size_synch_eFC)
effsize_pval[:, 1] = np.array(p_val_synch_eFC)

np.savetxt(r"D:\Shraddha\Data\synch_efc_eff_size_p_val_arbc.csv", effsize_pval, delimiter = ',')

