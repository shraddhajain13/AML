import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath
from sklearn.linear_model import LinearRegression
import pingouin as pg
from max_corr_sfc_efc_esc_all_atlas import gender_list_filtered

#model = 'Phase Oscillator Model'
model = 'LC Model'

sub_num_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
pheno_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
brain_size_list_full = pheno_data[:, 192] #list of all subject's brain sizes in the phenotypical data

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45'] 

l = len(atlas)

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

brain_size_list_ordered = rearr(brain_size_list_full)

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



if model == 'Phase Oscillator Model':

    #corr_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values #the file corr_sfc_efc_all_atlas_phase.csv has the value of corr(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.
    #corr_sfc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_phase.csv", header = None).values #the file corr_sfc_esc_all_atlas_phase.csv has the value of corr(sFC, eSC) arranged column wise for each atlas. One column has all the subjects.

    coup_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\coup_sfc_efc_all_atlas_phase.csv", header = None).values #the file coup_sfc_efc_all_atlas_phase.csv has the value of coup(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.

if model == 'LC Model':
    #corr_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values #the file corr_sfc_efc_all_atlas_lc.csv has the value of corr(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.
    #corr_sfc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_lc.csv", header = None).values #the file corr_sfc_esc_all_atlas_lc.csv has the value of corr(sFC, eSC) arranged column wise for each atlas. One column has all the subjects.

    coup_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\coup_sfc_efc_all_atlas_lc.csv", header = None).values #the file coup_sfc_efc_all_atlas_lc.csv has the value of corr(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.

residual_list_all_atlas = [] #will store all the residuals for all the parcellations column wise
male_data_all_atlas = []
female_data_all_atlas = []
p_val = []
eff_size = []

for i in range(l):
    #corr_efc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values[:, 1] #corr(eFC, eSC) for the given atlas
    
    #########...performing MLR...###########

    X = np.zeros([272, 1])
    X[:, 0] = np.array(brain_size_list_ordered)
    #X[:, 1] = np.array(np.arctanh(corr_efc_esc)) #with fisher z transform


    #Y = np.array(np.arctanh(corr_sfc_efc[:, i])) #corr(sFC, eFC) for the corresponding atlas (with fisher z transform)
    Y = np.array(coup_sfc_efc[:, i]) #coup_sfc_efc for the corresponding atlas

    reg = LinearRegression().fit(X, Y)
    coef = reg.coef_
    Y_hat = np.dot(X, np.array(coef)) #retaining the coefficient in regression analysis
    #print(len(Y_hat))
    #Y_hat_1 = reg.predict(X)
    #print(Y_hat_1 - Y_hat)
    

    residual = Y - Y_hat #list of 272 residual after regression of X without removing the constant

    ############...end of MLR...################


    #residual = np.tanh(residual) #inverse fisher z transform for box plots

    male_residual, female_residual = categorise_male_female(residual)

    male_data_all_atlas.append(male_residual) #male_data_all_atlas stores the residuals for males for all atlas column wise
    female_data_all_atlas.append(female_residual) #female_data_all_atlas stores the residuals for females for all atlas column wise

    #t_value, p_value = scipy.stats.ranksums(np.arctanh(male_residual), np.arctanh(female_residual)) #two tailed t test for corr(eFC, eSC)
    t_value, p_value = scipy.stats.ranksums(male_residual, female_residual) #for coupling strength
    
    p_val.append(p_value) #p_val is a list that stores the p value for residuals for all atlas
    eff_size.append(pg.compute_effsize(male_residual, female_residual, eftype = 'hedges')) # for coupling strength
    #eff_size.append(pg.compute_effsize(np.arctanh(male_residual), np.arctanh(female_residual), eftype = 'hedges')) #eff_size is a list that stores the effect size for residuals for all atlas
    

plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))
boxprops = {'linewidth': 2}
whiskerprops = {'linewidth': 2}
capprops = {'linewidth': 2}

male_plots = plt.boxplot(np.array(male_data_all_atlas).transpose(), positions = np.array(range(l))*2 - 0.3, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops) #the ones after regression
female_plots = plt.boxplot(np.array(female_data_all_atlas).transpose(), positions = np.array(range(l))*2 + 0.3, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops) #the ones after regression

male_data_br = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\male_data_coup_sfc_efc_lc_br.csv", header = None).values #loading the ones before reg
female_data_br = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\female_data_coup_sfc_efc_lc_br.csv", header = None).values #loading the ones before reg

male_plots_br = plt.boxplot(np.array(male_data_br), positions = np.array(range(l))*2 - 0.3, showfliers = False) #the ones before reg
female_plots_br = plt.boxplot(np.array(female_data_br), positions = np.array(range(l))*2 + 0.3, showfliers = False) #the ones before reg


set_box_color(male_plots, 'blue', 1) #after reg
set_box_color(female_plots, 'red', 1) #after reg

set_box_color(male_plots_br, 'grey', 0) #before reg
set_box_color(female_plots_br, 'grey', 0) #before reg

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.plot([], '--', color = 'grey', label = 'Before regression')
plt.legend(bbox_to_anchor=(1, 1.0), loc = 'upper left')

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 

plt.xticks(range(0, (l * 2), 2), atlas, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('Optimal coupling strength for corr(sFC, eFC)', fontsize = 20)
plt.tight_layout()
plt.show()


eff_size_p_val = np.zeros([l, 2])
eff_size_p_val[:, 0] = np.array(eff_size)
eff_size_p_val[:, 1] = np.array(p_val)

np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_coup_lc_arb.csv", eff_size_p_val, delimiter = ',')

