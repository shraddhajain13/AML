import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath
from max_corr_sfc_efc_esc_all_atlas import gender_list_filtered
import pingouin as pg
from MLR_all_atlas_sim import brain_size_list_ordered
from sklearn.linear_model import LinearRegression
#import outdated

#OUTDATED_IGNORE = 1

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

def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

eff_size = []
p_val = []

for i in range(l):
    data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values
    corr_efc_esc_list = data[:, 1]

    ####...MLR...####
    Y = corr_efc_esc_list
    X = np.zeros([272, 1])
    X[:, 0] = np.array(brain_size_list_ordered)

    reg = LinearRegression().fit(X, Y)
    Y_hat = reg.predict(X)
    residual = Y - Y_hat #list of 272 residual after regression of X
    ####...end of MLR...####

    corr_efc_esc_male, corr_efc_esc_female = categorise_male_female(residual) #replace 'residual' with corr_efc_esc_list if you do not want to regress anything

    t_value, p_value = scipy.stats.ranksums(corr_efc_esc_male, corr_efc_esc_female) #two tailed t test for corr(eFC, eSC)
    p_val.append(p_value)
    eff_size.append(pg.compute_effsize(corr_efc_esc_male, corr_efc_esc_female))

    male_data_all_atlas.append(corr_efc_esc_male)
    female_data_all_atlas.append(corr_efc_esc_female)
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
"""
eff_size_p_val_arr = np.zeros([l, 2])
eff_size_p_val_arr[:, 0] = eff_size
eff_size_p_val_arr[:, 1] = p_val
np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_arb.csv", eff_size_p_val_arr, delimiter = ',')

male_plots = plt.boxplot(np.array(male_data_all_atlas).transpose(), positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(np.array(female_data_all_atlas).transpose(), positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend()

plt.xticks(range(0, (l * 2), 2), atlas)
plt.xlim(-2, (l*2))
plt.title('Empirical correlations - After regression of brain size')
plt.xlabel('Atlas')
plt.ylabel('Corr(eFC, eSC)')
plt.show()
