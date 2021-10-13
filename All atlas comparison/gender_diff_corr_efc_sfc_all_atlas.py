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
from MLR_all_atlas_sim import brain_size_list_ordered, gender_list_filtered
from sklearn.linear_model import LinearRegression

x = np.linspace(1, 272, 272)

#this program is to make box plots for males and females for their values of corr(eFC, eSC) along with the wilcoxon sumrank test and cohen's D. 

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45'] 
l = len(atlas)
corr_efc_esc_male = []
corr_efc_esc_female = []

male_data_all_atlas = []
female_data_all_atlas = []

def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    list1 = [] #for males
    list2 = [] #for females
    for i in range(272):
        if(gender_list_filtered[i] == 'M'):
            list1.append(x[i])
        if(gender_list_filtered[i] == 'F'):
            list2.append(x[i])
    return list1, list2
r"""
plt.rcParams['font.size'] = '25'
plt.figure(figsize = (16, 8))
male_brain_list, female_brain_list = categorise_male_female(brain_size_list_ordered)
xm, xf = categorise_male_female(x)
plt.hist(male_brain_list, bins = 30, label = 'Male')
plt.hist(female_brain_list, bins = 30, label = 'Female')
#plt.xlabel('Subjects', fontsize = 25)
plt.xlabel('Brain size ($mm^{3}$)', fontsize = 25)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""

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

eff_size = []
p_val = []

for i in range(l):
    data = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values
    corr_efc_esc_list = data[:, 1]

    
    ####...MLR...####
    Y = np.arctanh(corr_efc_esc_list) #fisher z transform
    X = np.zeros([272, 1])
    X[:, 0] = np.array(brain_size_list_ordered)

    reg = LinearRegression().fit(X, Y)
    coef = reg.coef_
    Y_hat = np.dot(X, np.array(coef))
    residual = Y - Y_hat #list of 272 residual after regression of X
    ####...end of MLR...####
    
    residual = np.tanh(residual) #inverse fisher z transform
    corr_efc_esc_male, corr_efc_esc_female = categorise_male_female(residual) #replace 'residual' with corr_efc_esc_list if you do not want to regress anything
    
    male_data_all_atlas.append(corr_efc_esc_male)
    female_data_all_atlas.append(corr_efc_esc_female)

    t_value, p_value = scipy.stats.ranksums(np.arctanh(corr_efc_esc_male), np.arctanh(corr_efc_esc_female)) #two tailed t test for corr(eFC, eSC)
    p_val.append(p_value)
    eff_size.append(pg.compute_effsize(np.arctanh(corr_efc_esc_male), np.arctanh(corr_efc_esc_female)))

    
r"""
print('P value: ', p_val)
print('Effect size:', eff_size)

plt.plot(atlas, eff_size, marker = '.', markersize = 20, label = 'Effect size')
plt.plot(atlas, p_val, marker = '.', markersize = 20, label = 'Significance - p value')
plt.xlabel('Atlas')
plt.ylabel('Effect size (or) Significance')
plt.title('Effect size vs Atlas for Corr(eFC, eSC)')
plt.legend()
plt.show()

eff_size_p_val_arr = np.zeros([l, 2])
eff_size_p_val_arr[:, 0] = eff_size
eff_size_p_val_arr[:, 1] = p_val
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_arb.csv", eff_size_p_val_arr, delimiter = ',')

#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\male_data_efc_esc_emp_br.csv", np.array(male_data_all_atlas).transpose(), delimiter = ',')
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\female_data_efc_esc_emp_br.csv", np.array(female_data_all_atlas).transpose(), delimiter = ',')

male_data_br = pd.read_csv(r"D:\Shraddha\Data\male_data_efc_esc_emp_br.csv", header = None).values #loading the ones before reg
female_data_br = pd.read_csv(r"D:\Shraddha\Data\female_data_efc_esc_emp_br.csv", header = None).values #loading the ones before reg



plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))
boxprops = {'linewidth': 2}
whiskerprops = {'linewidth': 2}
capprops = {'linewidth': 2}

male_plots = plt.boxplot(np.array(male_data_all_atlas).transpose(), positions = np.array(range(l))*2 - 0.3, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops) #the ones after reg
female_plots = plt.boxplot(np.array(female_data_all_atlas).transpose(), positions = np.array(range(l))*2 + 0.3, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops) #the ones after reg

male_plots_br = plt.boxplot(np.array(male_data_br), positions = np.array(range(l))*2 - 0.3, showfliers = False) #the ones before reg
female_plots_br = plt.boxplot(np.array(female_data_br), positions = np.array(range(l))*2 + 0.3, showfliers = False) #the ones before reg

set_box_color(male_plots, 'blue', 1) 
set_box_color(female_plots, 'red', 1)

set_box_color(male_plots_br, 'grey', 0) 
set_box_color(female_plots_br, 'grey', 0)


plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.plot([], '--', c = 'grey', label = 'Before regression')
plt.legend(loc = 'upper left')

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 
plt.xticks(range(0, (l * 2), 2), atlas, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('corr(eFC, eSC)', fontsize = 20)
plt.tight_layout()
plt.savefig(r"C:\Users\shrad\Desktop\corr_efc_esc_box_plot.png", dpi = 300)
plt.show()
"""