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


#model = 'Phase Oscillator Model'
model = 'LC Model'
pheno_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
sub_num_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))


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

def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

a = 0

if model == 'Phase Oscillator Model': #checking which model it is
    path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase"
    parcel_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\list_of_parcellations_phase.txt", dtype = str)
    ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25CPU', 'HO25GPU', 'HO35', 'HO45']

if model == 'LC Model':
    path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\LC"
    parcel_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\list_of_parcellations_lc.txt", dtype = str)
    ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45']

l = len(ticks)
female_data_sfc_efc = np.zeros([144, l]) #144 females
male_data_sfc_efc = np.zeros([128, l]) #128 males

female_data_sfc_esc = np.zeros([144, l])
male_data_sfc_esc = np.zeros([128, l])

eff_size_sfc_efc = []
eff_size_sfc_esc = []

corr_sfc_efc_all_atlas = []
corr_sfc_esc_all_atlas = []

p_val = []
for filename in parcel_list: #looping over parcels
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

    #print('Male: ', len(corr_sfc_efc_male))
    #print('Female: ', len(corr_sfc_efc_female))
    
    male_data_sfc_efc[:, a] = corr_sfc_efc_male
    female_data_sfc_efc[:, a] = corr_sfc_efc_female

    male_data_sfc_esc[:, a] = corr_sfc_esc_male
    female_data_sfc_esc[:, a] = corr_sfc_esc_female

    corr_sfc_efc_all_atlas.append(corr_sfc_efc_list) #making a list of lists - 2D list that has corr(sFC, eFC) for all atlases
    corr_sfc_esc_all_atlas.append(corr_sfc_esc_list) #making a list of lists - 2D list that has corr(sFC, eSC) for all atlases
    
    a = a + 1
    
    t_value_fc, p_value_fc = scipy.stats.ranksums(corr_sfc_efc_male, corr_sfc_efc_female) #two tailed t test for corr(sFC, eFC)
    t_value_sc, p_value_sc = scipy.stats.ranksums(corr_sfc_esc_male, corr_sfc_esc_female) #two tailed t test for corr(sFC, eSC)
    eff_size_sfc_efc.append(pg.compute_effsize(corr_sfc_efc_male, corr_sfc_efc_female))
    eff_size_sfc_esc.append(pg.compute_effsize(corr_sfc_esc_male, corr_sfc_esc_female))

    p_val.append(p_value_fc)
    #print(filename)
    #print(corr_sfc_efc_list)
    #print('corr(sFC, eFC): ', t_value_fc, p_value_fc)
    #print('corr(sFC, eSC): ', t_value_sc, p_value_sc)

r"""
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))

male_plots = plt.boxplot(male_data_sfc_efc, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(female_data_sfc_efc, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend()

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 45)
plt.xlim(-2, (l*2))
#plt.title(model + " - Before Regression")
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('Corr(sFC, eFC)', fontsize = 20)
plt.tight_layout()
#plt.savefig(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Plots\New folder\corr_sfc_efc_all_atlas_phase_before_reg.png")
plt.show()

"""
#print('Maximum effect size corr(sFC, eFC) across atlases: ', max(eff_size_sfc_efc), ticks[eff_size_sfc_efc.index(max(eff_size_sfc_efc))])
#print('Minimum effect size corr(sFC, eFC) across atlases: ', min(eff_size_sfc_efc), ticks[eff_size_sfc_efc.index(min(eff_size_sfc_efc))])
#print('Maximum effect size corr(sFC, eSC) across atlases: ', max(eff_size_sfc_esc), ticks[eff_size_sfc_esc.index(max(eff_size_sfc_esc))])
#print('Minimum effect size corr(sFC, eSC) across atlases: ', min(eff_size_sfc_esc), ticks[eff_size_sfc_esc.index(min(eff_size_sfc_esc))])

#plt.plot(ticks, eff_size_sfc_efc,'.', label = 'Cohens D - corr(sFC, eFC)')
#plt.plot(ticks, eff_size_sfc_esc, '.', label = 'Cohens D - corr(sFC, eSC)')

#plt.title('Effect sizes of different atlases - LC Model')
#plt.ylabel('Effect size')
#plt.xlabel('Atlas')
#plt.legend()
#plt.show()
#print(corr_sfc_efc_all_atlas)
#print(corr_sfc_esc_all_atlas)
#print(eff_size_sfc_efc)
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_lc.csv", np.array(corr_sfc_efc_all_atlas).transpose(), delimiter = ',')
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_lc.csv", np.array(corr_sfc_esc_all_atlas).transpose(), delimiter = ',')

#eff_size_p_val = np.zeros([l, 2])
#eff_size_p_val[:, 0] = np.array(eff_size_sfc_efc)
#eff_size_p_val[:, 1] = np.array(p_val)
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_lc_br.csv", eff_size_p_val, delimiter = ',')

#print(model)
#print(corr_sfc_efc_all_atlas)
#print('Effect size Corr(sFC, eFC) before regression:', eff_size_sfc_efc)
#print('P value corr(sFC, eFC) before regression:', p_val)
#print('Effect size Corr(sFC, eSC):', eff_size_sfc_esc)
