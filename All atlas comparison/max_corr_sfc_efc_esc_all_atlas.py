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
pheno_data = pd.read_csv(r"E:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
sub_num_list = np.loadtxt(r"E:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))


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
ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45']

if model == 'Phase Oscillator Model': #checking which model it is
    parcel_list = np.loadtxt(r"E:\Shraddha\Data\list_of_parcellations_phase.txt", dtype = str)
    

if model == 'LC Model':
    parcel_list = np.loadtxt(r"E:\Shraddha\Data\list_of_parcellations_lc.txt", dtype = str)


l = len(ticks)
#female_data_sfc_efc = np.zeros([144, l]) #144 females
#male_data_sfc_efc = np.zeros([128, l]) #128 males

#female_data_sfc_esc_func_fit = np.zeros([144, l]) #144 females
#male_data_sfc_esc_func_fit = np.zeros([128, l]) #128 males

female_data_sfc_efc_struc_fit = np.zeros([144, l]) #144 females
male_data_sfc_efc_struc_fit = np.zeros([128, l]) #128 males

#female_data_fc_coup = np.zeros([144, l])
#male_data_fc_coup = np.zeros([128, l])

#female_data_sfc_esc = np.zeros([144, l]) #model fitting - extracting the sFC that is best correlated with eSC
#male_data_sfc_esc = np.zeros([128, l])

#eff_size_sfc_efc = []
#eff_size_sfc_esc = []
#eff_size_fc_coup = []

#corr_sfc_efc_all_atlas = []
#corr_sfc_esc_func_fit_all_atlas = []
#corr_sfc_esc_all_atlas = [] #this will store the corr(sFC, eSC) for all atlases column wise
corr_sfc_efc_struc_fit_all_atlas = [] #this will store corr(sfc, efc) - struc fit for all atlases column wise

#coup_sfc_efc_all_atlas = []
#coup_sfc_efc_all_atlas = []

p_val = []
#eff_size_sfc_esc = []
#eff_size_sfc_esc_func_fit = []
eff_size_sfc_efc_struc_fit = []

for filename in parcel_list: #looping over parcels
    #data = np.loadtxt(filename, usecols = (29, 30, 31, 32, 33, 34))
    #vals = np.loadtxt(filename, usecols = (4)) #for corr(sfc esc) func fit
    vals = np.loadtxt(filename, usecols = (11)) #for corr(sfc efc) struc fit
    k = 0
    #delay_fc_list = []
    #delay_sc_list = []
    #coup_str_fc_list = []
    #coup_str_sc_list = []
    #corr_sfc_efc_list = []
    #corr_sfc_esc_list = [] #extracting the list of best fit corr(sFC, eSC) for each atlas
    #corr_sfc_esc_func_fit_list = []
    corr_sfc_efc_struc_fit_list = [] #collects the corr(sfc, efc) - struc fit for this given atlas and is reset to zero for the next atlas

    for i in range(272): #loop to extract the goodness of fit for each subject. It is the first row for each subject in the file of each parcellation.
        
        #delay_sfc_efc = data[k, 0]
        #coup_str_sfc_efc = data[k, 1]
        #corr_sfc_efc = data[k, 2]

        #corr_sfc_esc_func_fit = vals[k]
        corr_sfc_efc_struc_fit = vals[k]

        #delay_sfc_esc = data[k, 3]
        #coup_str_sfc_esc = data[k, 4]
        #corr_sfc_esc = data[k, 5]

        #delay_fc_list.append(delay_sfc_efc)
        #delay_sc_list.append(delay_sfc_esc)

        #coup_str_fc_list.append(coup_str_sfc_efc)
        #coup_str_sc_list.append(coup_str_sfc_esc)

        #corr_sfc_efc_list.append(corr_sfc_efc)
        #corr_sfc_esc_func_fit_list.append(corr_sfc_esc_func_fit)
        corr_sfc_efc_struc_fit_list.append(corr_sfc_efc_struc_fit)

        k = k + 25

    #delay_male, delay_female = categorise_male_female(delay_list)
    #coup_str_male, coup_str_female = categorise_male_female(coup_str_fc_list)


    #corr_sfc_efc_male, corr_sfc_efc_female = categorise_male_female(corr_sfc_efc_list) #splitting into males and females
    #corr_sfc_esc_male, corr_sfc_esc_female = categorise_male_female(corr_sfc_esc_list) #splitting into males and females
    #corr_sfc_esc_func_fit_male, corr_sfc_esc_func_fit_female = categorise_male_female(corr_sfc_esc_func_fit_list)
    corr_sfc_efc_struc_fit_male, corr_sfc_efc_struc_fit_female = categorise_male_female(corr_sfc_efc_struc_fit_list)

    #print('Male: ', len(corr_sfc_efc_male))
    #print('Female: ', len(corr_sfc_efc_female))

    #male_data_fc_coup[:, a] = coup_str_male
    #female_data_fc_coup[:, a] = coup_str_female
    
    #male_data_sfc_efc[:, a] = corr_sfc_efc_male
    #female_data_sfc_efc[:, a] = corr_sfc_efc_female

    #male_data_sfc_esc[:, a] = corr_sfc_esc_male
    #female_data_sfc_esc[:, a] = corr_sfc_esc_female

    #male_data_sfc_esc_func_fit[:, a] = corr_sfc_esc_func_fit_male
    #female_data_sfc_esc_func_fit[:, a] = corr_sfc_esc_func_fit_female

    male_data_sfc_efc_struc_fit[:, a] = corr_sfc_efc_struc_fit_male
    female_data_sfc_efc_struc_fit[:, a] = corr_sfc_efc_struc_fit_female

    #corr_sfc_efc_all_atlas.append(corr_sfc_efc_list) #making a list of lists - 2D list that has corr(sFC, eFC) for all atlases stored column wise
    corr_sfc_efc_struc_fit_all_atlas.append(corr_sfc_efc_struc_fit_list) #making a list of lists - 2D list that has corr(sFC, eSC) for all atlases stored column wise

    #corr_sfc_esc_func_fit_all_atlas.append(corr_sfc_esc_func_fit_list) #making a list of lists - 2D list that has corr(sFC, eSC) - sFC is from functional fitfor all atlases stored column wise
    #coup_sfc_efc_all_atlas.append(coup_str_fc_list) #making a list of lists - 2D list that has coup_str(sFC, eFC) for all atlases stored column wise
    
    a = a + 1
    
    #t_value_fc, p_value_fc = scipy.stats.ranksums(np.arctanh(corr_sfc_efc_male), np.arctanh(corr_sfc_efc_female)) #two tailed t test for corr(sFC, eFC)
    #t_value_sc, p_value_sc = scipy.stats.ranksums(np.arctanh(corr_sfc_esc_male), np.arctanh(corr_sfc_esc_female)) #two tailed t test for corr(sFC, eSC)
    #eff_size_sfc_efc.append(pg.compute_effsize(np.arctanh(corr_sfc_efc_male), np.arctanh(corr_sfc_efc_female), eftype = 'hedges'))

    #eff_size_sfc_esc.append(pg.compute_effsize(np.arctanh(corr_sfc_esc_male), np.arctanh(corr_sfc_esc_female), eftype = 'hedges'))

    #t_value, p_value = scipy.stats.ranksums(np.arctanh(corr_sfc_esc_func_fit_male), np.arctanh(corr_sfc_esc_func_fit_female))
    #eff_size_sfc_esc_func_fit.append(pg.compute_effsize(np.arctanh(corr_sfc_esc_func_fit_male), np.arctanh(corr_sfc_esc_func_fit_female), eftype = 'hedges')) #effect size for coupling strength

    t_value, p_value = scipy.stats.ranksums(np.arctanh(corr_sfc_efc_struc_fit_male), np.arctanh(corr_sfc_efc_struc_fit_female))
    eff_size_sfc_efc_struc_fit.append(pg.compute_effsize(np.arctanh(corr_sfc_efc_struc_fit_male), np.arctanh(corr_sfc_efc_struc_fit_female), eftype = 'hedges')) #effect size for coupling strength
    p_val.append(p_value)
    
r"""
np.savetxt(r"E:\Shraddha\Data\corr_sfc_efc_struc_fit_all_atlas_lc.csv", np.array(corr_sfc_efc_struc_fit_all_atlas).transpose(), delimiter = ',') #saving the corr(sFC, eSC) - functional fit for all atlas where each column represents one atlas in the .csv file


plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))

male_plots = plt.boxplot(male_data_sfc_efc_struc_fit, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(female_data_sfc_efc_struc_fit, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

ticks = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] #only for the purpose of x axis

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Atlas', fontsize = 20)
plt.ylabel('corr(sFC, eFC) - sFC is from structural fitting', fontsize = 20)
plt.tight_layout()
plt.show()




#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\male_data_coup_sfc_efc_lc_br.csv", male_data_fc_coup, delimiter = ',') #this storage is for the grey plots in the background
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\female_data_coup_sfc_efc_lc_br.csv", female_data_fc_coup, delimiter = ',') #this storage is for the grey plots in the background

#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_efc_all_atlas_lc.csv", np.array(corr_sfc_efc_all_atlas).transpose(), delimiter = ',')
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_sfc_esc_all_atlas_lc.csv", np.array(corr_sfc_esc_all_atlas).transpose(), delimiter = ',')

#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\male_data_sfc_efc_lc_br.csv", male_data_sfc_efc, delimiter = ',') #this storage is for the grey plots in the background
#np.savetxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\female_data_sfc_efc_lc_br.csv", female_data_sfc_efc, delimiter = ',') #this storage is for the grey plots in the background



np.savetxt(r"E:\Shraddha\Data\male_data_sfc_efc_struc_fit_lc_br.csv", male_data_sfc_efc_struc_fit, delimiter = ',') #this storage is for the grey plots in the background
np.savetxt(r"E:\Shraddha\Data\female_data_sfc_efc_struc_fit_lc_br.csv", female_data_sfc_efc_struc_fit, delimiter = ',') #this storage is for the grey plots in the background

eff_size_p_val = np.zeros([l, 2])
eff_size_p_val[:, 0] = np.array(eff_size_sfc_efc_struc_fit)
eff_size_p_val[:, 1] = np.array(p_val)
np.savetxt(r"E:\Shraddha\Data\eff_size_p_val_corr_sfc_efc_struc_fit_lc_br.csv", eff_size_p_val, delimiter = ',')
"""