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


path = r"D:\Shraddha\std_dev_all_atlas"
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

def set_box_color(bp, color): #setting color for box plots
    #if flag == 1:
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

    #if flag == 0:
        #plt.setp(bp['boxes'], color=color, linestyle = '--')
        #plt.setp(bp['whiskers'], color=color, linestyle = '--')
        #plt.setp(bp['caps'], color=color, linestyle = '--')
        #plt.setp(bp['medians'], color=color, linestyle = '--')


std_eFC_male_all_atlas = np.zeros([128, 11])
std_eFC_female_all_atlas = np.zeros([144, 11])

#mean_std_time_series_male_all_atlas = np.zeros([128, 11])
#mean_std_time_series_female_all_atlas = np.zeros([144, 11])

#std_of_std_time_series_male_all_atlas = np.zeros([128, 11])
#std_of_std_time_series_female_all_atlas = np.zeros([144, 11])

eff_size_std_eFC = []
p_val_std_eFC = []

#eff_size_std_time_series = []
#p_val_std_time_series = []

#corr_sfc_efc_phase = pd.read_csv(r"E:\Shraddha\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values
#corr_sfc_efc_lc = pd.read_csv(r"E:\Shraddha\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values

#corr_sfc_esc_best_fit_phase = pd.read_csv(r"E:\Shraddha\Data\corr_sfc_esc_all_atlas_phase.csv", header = None).values
#corr_sfc_esc_best_fit_lc = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_esc_all_atlas_lc.csv", header = None).values


for i in range(len(atlas)):
    std_eFC = pd.read_csv(os.path.join(path, "std_eFC_"+ atlas[i]) + '.csv').values[:, 1]
    #corr_efc_esc = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values[:, 1] #corr(eFC, eSC) for the given atlas
    #mean_std_time_series = pd.read_csv(os.path.join(path, "mean_std_time_series_"+ atlas[i]) + '.csv').values[:, 1]
    #std_of_std_time_series = pd.read_csv(os.path.join(path, "std_of_std_time_series_"+ atlas[i]) + '.csv').values[:, 1]
    #corr_efc_esc = pd.read_csv(r"E:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + '.csv', header = None).values[:, 1]
    #print(corr_efc_esc)
    #print(std_eFC)

    #pearson_corr = stats.pearsonr(std_eFC, corr_sfc_esc_best_fit_lc[:, i])[0]
    r"""
    ##########.....MLR....############
    
    X = np.zeros([272, 2])
    X[:, 0] = np.array(brain_size_list_sorted)
    X[:, 1] = np.array(np.arctanh(corr_efc_esc))

    Y = std_eFC

    reg = LinearRegression().fit(X, Y)
    coef = reg.coef_
    Y_hat = np.dot(X, np.array(coef)) #retaining the costant in regression analysis

    residual = Y - Y_hat

    #resdiual = np.tanh(residual)

    #########.....end of MLR....#########
    """
    
    std_eFC_male, std_eFC_female = categorise_male_female(std_eFC)
    std_eFC_male_all_atlas[:, i] = np.array(std_eFC_male)
    std_eFC_female_all_atlas[:, i] = np.array(std_eFC_female)
    t_value1, p_value1 = scipy.stats.ranksums(std_eFC_male, std_eFC_female)
    eff_size_std_eFC.append(pg.compute_effsize(std_eFC_male, std_eFC_female, eftype = 'hedges')) #effect size for coupling strength
    p_val_std_eFC.append(p_value1)

    #mean_std_time_series_male, mean_std_time_series_female = categorise_male_female(mean_std_time_series)
    #std_time_series_male_all_atlas[:, i] = np.array(mean_std_time_series_male)
    #std_time_series_female_all_atlas[:, i] = np.array(mean_std_time_series_female)
    #t_value2, p_value2 = scipy.stats.ranksums(mean_std_time_series_male, mean_std_time_series_female)
    #eff_size_std_time_series.append(pg.compute_effsize(mean_std_time_series_male, mean_std_time_series_female, eftype = 'hedges')) #effect size for coupling strength
    #p_val_std_time_series.append(p_value2)

    #std_of_std_time_series_male, std_of_std_time_series_female = categorise_male_female(std_of_std_time_series)
    #std_of_std_time_series_male_all_atlas[:, i] = std_of_std_time_series_male
    #std_of_std_time_series_female_all_atlas[:, i] = std_of_std_time_series_female


    #corr_sfc_efc_phase_male, corr_sfc_efc_phase_female = categorise_male_female(corr_sfc_efc_phase[:, i])
    #corr_sfc_efc_lc_male, corr_sfc_efc_lc_female = categorise_male_female(corr_sfc_efc_lc[:, i])

    #corr_sfc_esc_best_fit_phase_male, corr_sfc_esc_best_fit_phase_female = categorise_male_female(corr_sfc_esc_best_fit_phase[:, i])
    #corr_sfc_esc_best_fit_lc_male, corr_sfc_esc_best_fit_lc_female = categorise_male_female(corr_sfc_esc_best_fit_lc[:, i])

    
    
    r"""
    plt.rcParams['font.size'] = '20'
    plt.figure(figsize = (16, 8))
    plt.text(max(corr_sfc_esc_best_fit_lc[:, i])*0.9, max(std_eFC)*0.99, 'Pearsons corr = ' + str(round(pearson_corr, 2)), size = 20)#, ha = 'center', va = 'center')
    plt.plot(corr_sfc_esc_best_fit_lc_male, std_eFC_male, '.', label = 'Male')
    plt.plot(corr_sfc_esc_best_fit_lc_female, std_eFC_female, '.', label = 'Female')
    plt.title(atlas[i] + ' - Limit Cycle Model')
    plt.xlabel('corr(sFC, eSC) - Best Fit', fontsize = 20)
    plt.ylabel('Std of eFC', fontsize = 20)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig(r"E:\Shraddha\Plots\Data Variables\Std_eFC\corr_sfc_esc_best_fit_vs_std_efc\LC\std_eFC_vs_corr_sfc_esc_lc_" + atlas[i] + '.png')
    plt.show()
    """
    
np.savetxt(r"D:\Shraddha\Data\male_data_std_efc_br.csv", std_eFC_male_all_atlas, delimiter = ',') #this storage is for the grey plots in the background
np.savetxt(r"D:\Shraddha\Data\female_data_std_efc_br.csv", std_eFC_female_all_atlas, delimiter = ',') #this storage is for the grey plots in the background

l = len(atlas)
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))


#male_data_br = pd.read_csv(r"D:\Shraddha\Data\male_data_std_efc_br.csv", header = None).values #loading the ones before reg
#female_data_br = pd.read_csv(r"D:\Shraddha\Data\female_data_std_efc_br.csv", header = None).values #loading the ones before reg

#male_plots_br = plt.boxplot(np.array(male_data_br), positions = np.array(range(l))*2 - 0.3, showfliers = False) #the ones before reg
#female_plots_br = plt.boxplot(np.array(female_data_br), positions = np.array(range(l))*2 + 0.3, showfliers = False) #the ones before reg

male_plots = plt.boxplot(std_eFC_male_all_atlas, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(std_eFC_female_all_atlas, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue')#, 1) 
set_box_color(female_plots, 'red')#, 1)

#set_box_color(male_plots_br, 'grey', 0) #before reg
#set_box_color(female_plots_br, 'grey', 0) #before reg

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
#plt.plot([], '--', color = 'grey', label = 'Before regression')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] #only for the purpose of x axis

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('Std dev of eFC', fontsize = 20)
plt.tight_layout()
plt.show()

effsize_pval = np.zeros([l, 2])
effsize_pval[:, 0] = np.array(eff_size_std_eFC)
effsize_pval[:, 1] = np.array(p_val_std_eFC)

np.savetxt(r"D:\Shraddha\Data\std_efc_eff_size_p_val_br.csv", effsize_pval, delimiter = ',')

r"""

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 

p_val_std_eFC = multitest.fdrcorrection(np.array(p_val_std_eFC))[1]
#p_val_std_time_series = multitest.fdrcorrection(np.array(p_val_std_time_series))[1]
plt.rcParams['font.size'] = '20'
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (16, 9))
ax1.plot(np.array(atlas), np.array(eff_size_std_eFC), marker = '.', markersize = 10)
ax1.yaxis.grid()
ax1.set_title('Hedges g vs Atlas')
ax1.set_ylabel('Hedges g', fontsize = 20)

ax2.plot(np.array(atlas), np.array(p_val_std_eFC), marker = '.', markersize = 10)
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

#print(eff_size_std_eFC)
#print(multitest.fdrcorrection(np.array(p_val_std_eFC))[1])

"""