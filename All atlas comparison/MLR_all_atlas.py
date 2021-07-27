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

model = 'Phase Oscillator Model'
#model = 'LC Model'

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


def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if model == 'Phase Oscillator Model':

    corr_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values #the file corr_sfc_efc_all_atlas.csv has the value of corr(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.
    corr_sfc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_phase.csv", header = None).values #the file corr_sfc_esc_all_atlas.csv has the value of corr(sFC, eSC) arranged column wise for each atlas. One column has all the subjects.

    corr_sfc_efc = np.delete(corr_sfc_efc, 9, 1)
    corr_sfc_esc = np.delete(corr_sfc_esc, 9, 1)


if model == 'LC Model':
    corr_sfc_efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values #the file corr_sfc_efc_all_atlas.csv has the value of corr(sFC, eFC) arranged column wise for each atlas. One column has all the subjects.
    corr_sfc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_lc.csv", header = None).values #the file corr_sfc_esc_all_atlas.csv has the value of corr(sFC, eSC) arranged column wise for each atlas. One column has all the subjects.

#print(corr_sfc_efc.shape)


residual_list_all_atlas = [] #will store all the residuals for all the parcellations column wise
male_data_all_atlas = []
female_data_all_atlas = []
p_val = []
eff_size = []

for i in range(l):
    corr_efc_esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Empirical_correlations_all_atlas\Atlas_" + atlas[i] + ".csv", header = None).values[:, 1] #corr(eFC, eSC) for the given atlas
    #print(corr_efc_esc)
    
    #########...performing MLR...###########

    X = np.zeros([272, 2])
    X[:, 0] = np.array(brain_size_list_ordered)
    X[:, 1] = np.array(corr_efc_esc)

    Y = np.array(corr_sfc_efc[:, i]) #corr(sFC, eFC) for the corresponding atlas

    reg = LinearRegression().fit(X, Y)
    Y_hat = reg.predict(X)
    residual = Y - Y_hat #list of 272 residual after regression of X

    ############...end of MLR...################

    residual_list_all_atlas.append(residual)  #residual_list_all_atlas stores the residuals for all atlas column wise

    male_residual, female_residual = categorise_male_female(residual)

    male_data_all_atlas.append(male_residual) #male_data_all_atlas stores the residuals for males for all atlas column wise
    female_data_all_atlas.append(female_residual) #female_data_all_atlas stores the residuals for females for all atlas column wise

    t_value, p_value = scipy.stats.ranksums(male_residual, female_residual) #two tailed t test for corr(eFC, eSC)
    p_val.append(p_value) #p_val is a list that stores the p value for residuals for all atlas
    eff_size.append(pg.compute_effsize(male_residual, female_residual)) #eff_size is a list that stores the effect size for residuals for all atlas


male_plots = plt.boxplot(np.array(male_data_all_atlas).transpose(), positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(np.array(female_data_all_atlas).transpose(), positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend()

plt.xticks(range(0, (l * 2), 2), atlas)
plt.xlim(-2, (l*2))
plt.title('Phase Oscillator Model - After regression of brain size and corr(eFC, eSC)')
plt.xlabel('Atlas')
plt.ylabel('Residuals')
plt.show()

print("Effect size after regressing both brain size and corr(eFC, eSC):", eff_size)
print("P value after regressing both brain size and corr(eFC, eSC):", p_val)