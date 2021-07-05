
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import ntpath
import os
import pandas as pd
import math
from statistics import mean

data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv") ##for phenotypical data
#print(data.shape)
sub_num_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
sub_num = data.iloc[:,0]

sub_num_list_phen = sub_num.values.tolist() #getting the list of subject number from phenotypical data
gender = data.iloc[:,3]
gender_list = gender.values.tolist() #getting list of genders from phenotypical data


path = r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Ph_Osc_Schaefer100_2Dim_par\Ph_Osc_Schaefer100_2Dim_par'
max_corr_list_fc = []
max_corr_list_sc = []
#min_corr_list = []
max_delay_list_fc = []
max_delay_list_sc = []
max_coup_str_list_fc = []
max_coup_str_list_sc = []
#min_delay_list = []
#min_coup_str_list = []
#corr_3_4_list = []
subject_number_list_sim = [] #list of subject number from the simulated data

for filename in glob.glob(os.path.join(path, '*bif_all')):
    #print(filename)
    
    subject_number = (int)(ntpath.basename(filename)[0:6])
    print(subject_number)
    subject_number_list_sim.append(subject_number)
    array_txt = np.loadtxt(filename,usecols=(0, 1, 2,3))
    delay = array_txt[:,0]
    coup_str = array_txt[:,1]
    corr_simfn_empfn = array_txt[:,2]
    corr_simfn_empsc = array_txt[:,3]
    max_corr_fc = max(corr_simfn_empfn)
    max_corr_sc = max(corr_simfn_empsc)
    #min_corr = min(corr_simfn_empfn)
    print(max_corr_fc)
    max_corr_list_fc.append(max_corr_fc) #bestfit corr(sfc, efc)
    max_corr_list_sc.append(max_corr_sc) #bestfit corr(sfc, esc)
    #min_corr_list.append(min_corr)
    
    max_corr_index_fc = np.argmax(corr_simfn_empfn)
    max_corr_index_sc = np.argmax(corr_simfn_empsc)
    #min_corr_index = np.argmin(corr_simfn_empfn)
    
    max_delay_list_fc.append(delay[max_corr_index_fc]) #tau for bestfit corr(sfc, efc)
    max_delay_list_sc.append(delay[max_corr_index_sc]) #tau for bestfit corr(sfc, esc)
    #min_delay_list.append(delay[min_corr_index])
    
    max_coup_str_list_fc.append(coup_str[max_corr_index_fc]) #coupling strength for bestfit corr(sfc, efc)
    max_coup_str_list_sc.append(coup_str[max_corr_index_sc]) #coupling strength for bestfit corr(sfc, esc)
    #min_coup_str_list.append(coup_str[min_corr_index])
    
    break
#corr_3_4, _ = stats.pearsonr(np.array(max_corr_list_3), np.array(max_corr_list_4))
    #corr_3_4_list.append(corr_3_4)
    
    
#subnum_vs_corr_empfn = np.zeros([272,2])
#subnum_vs_corr_empfn[:,0] = np.array(subject_number_list_sim)
#subnum_vs_corr_empfn[:,1] = np.array(max_corr_list_3)
#print(subnum_vs_corr_empfn)
#print("List of maximum correlation = ", max_corr_list)
#print("List of minimum correlation = ", min_corr_list)
#print("List of delays for maximum correlation = ", max_delay_list)
#print("List of coupling strength for maximum correlation = ", max_coup_str_list)
#print("List of delays for minimum correlation = ", min_delay_list) 
#print("List of coupling strength for minimum correlation = ", min_coup_str_list)
#print("List of pearson correlation between column 3 and 4 =", corr_3_4_list)
#print(corr_3_4)
#print(np.array(max_corr_list_3))


def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    return_list1 = [] #for males
    return_list2 = [] #for females
    for i in range(len(subject_number_list_sim)):
        index = sub_num_list_phen.index(subject_number_list_sim[i])
        gen = gender_list[index]
        if(gen == 'M'):
            return_list1.append(x[i])
        if(gen == 'F'):
            return_list2.append(x[i])
    return return_list1, return_list2
        
male_fc_list, female_fc_list = categorise_male_female(max_corr_list_fc)
male_sc_list, female_sc_list = categorise_male_female(max_corr_list_sc)
male_delay_fc, female_delay_fc = categorise_male_female(max_delay_list_fc)
male_delay_sc, female_delay_sc = categorise_male_female(max_delay_list_sc)
male_coup_str_fc, female_coup_str_fc = categorise_male_female(max_coup_str_list_fc)
male_coup_str_sc, female_coup_str_sc = categorise_male_female(max_coup_str_list_sc)
#print(len(male_fc_list))
#print(len(female_fc_list))
#t_value_fc, p_value_fc = scipy.stats.ttest_ind(male_fc_list, female_fc_list, alternative='greater')
#t_value_sc, p_value_sc = scipy.stats.ttest_ind(male_sc_list, female_sc_list, alternative = 'less')
#t_delay_fc, p_delay_fc = scipy.stats.ttest_ind(male_delay_fc, female_delay_fc) #no difference in decision making for less or greater
#t_delay_sc, p_delay_sc = scipy.stats.ttest_ind(male_delay_sc, female_delay_sc)
#t_coup_fc, p_coup_fc = scipy.stats.ttest_ind(male_coup_str_fc, female_coup_str_fc, alternative = 'greater')
#t_coup_sc, p_coup_sc = scipy.stats.ttest_ind(male_coup_str_sc, female_coup_str_sc)

t_value_fc, p_value_fc = scipy.stats.ranksums(male_fc_list, female_fc_list)
t_value_sc, p_value_sc = scipy.stats.ranksums(male_sc_list, female_sc_list)
t_delay_fc, p_delay_fc = scipy.stats.ranksums(male_delay_fc, female_delay_fc) 
t_delay_sc, p_delay_sc = scipy.stats.ranksums(male_delay_sc, female_delay_sc)
t_coup_fc, p_coup_fc = scipy.stats.ranksums(male_coup_str_fc, female_coup_str_fc)
t_coup_sc, p_coup_sc = scipy.stats.ranksums(male_coup_str_sc, female_coup_str_sc)
r"""
print("t and p value for correlation FC = ", t_value_fc, p_value_fc)
print("t and p value for correlation SC = ", t_value_sc, p_value_sc)
print("t and p value for best fit delay FC = ", t_delay_fc, p_delay_fc)
print("t and p value for best fit delay SC = ", t_delay_sc, p_delay_sc)
print("t and p value for best fit coupling strength FC = ", t_coup_fc, p_coup_fc)
print("t and p value for best fit coupling strength SC = ", t_coup_sc, p_coup_sc)

fc_data = [male_fc_list, female_fc_list]
sc_data = [male_sc_list, female_sc_list]
fig, ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].boxplot(fc_data)
ax[0].set_xticklabels(['Male','Female'])
ax[0].set_ylabel('Best fit correlation between sFC and eFC')
ax[0].set_title('FC')
ax[1].boxplot(sc_data)
ax[1].set_xticklabels(['Male','Female'])
ax[1].set_ylabel('Best fit correlation between sFC and eSC')
ax[1].set_title('SC')
plt.show()
"""
r"""
def cohens_D(list1, list2): #function for finding the Cohen's D
    var1 = np.var(list1)
    var2 = np.var(list2)
    n1 = len(list1)
    n2 = len(list2)
    x1 = mean(list1)
    x2 = mean(list2)
    pooled_stdev = math.sqrt((((n1-1)*var1) + ((n2-1)*var2))/(n1+n2-1))
    D = (x1-x2)/pooled_stdev
    return D



D_fc = cohens_D(male_fc_list, female_fc_list)
D_sc = cohens_D(male_sc_list, female_sc_list)
print("Cohen's D for FC = ", D_fc)
print("Cohen's D for SC = ", D_sc)
plt.hist([male_sc_list, female_sc_list], label = ['male', 'female'])
plt.legend()
plt.show()
"""
def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(subject_number_list_sim == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

corr_sfc_efc_list = rearr(max_corr_list_fc)
#print(max_corr_list_fc)
#print(corr_sfc_efc_list)
#print(rearr(subject_number_list_sim))
#print(sub_num_list)