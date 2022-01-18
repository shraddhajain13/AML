import numpy as np
import pandas as pd
import os
import random


r"""
efc = pd.read_csv(r"D:\Shraddha\FC_matrices_all_atlas\efc_S200.csv", header = None).values
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]
#brain_size_list_full = pheno_data[:, 192]

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list_old[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

gender_list_filtered = rearr(gender_list)
#brain_size_list_sorted = rearr(brain_size_list_full)

def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    list1 = [] #for males
    list2 = [] #for females 
    for i in range(272):
        if(gender_list_filtered[i] == 'M'):
            list1.append(x[i])
        if(gender_list_filtered[i] == 'F'):
            list2.append(x[i])
    return list1, list2

efc_male, efc_female = categorise_male_female(efc)
sub_list_male, sub_list_female = categorise_male_female(sub_num_list_old)
#np.savetxt(r"D:\Shraddha\FC_matrices_separated_male_female\S200_male.csv", (np.array(efc_male)).transpose(), delimiter = ',')
#np.savetxt(r"D:\Shraddha\FC_matrices_separated_male_female\S200_female.csv", (np.array(efc_female)).transpose(), delimiter = ',')
#print((np.array(efc_male).transpose())[:, 0])
#print((np.array(efc_female).transpose())[:, 0])
np.savetxt(r"D:\Shraddha\Data\sub_list_male.csv", np.array(sub_list_male), delimiter = ',')
np.savetxt(r"D:\Shraddha\Data\sub_list_female.csv", np.array(sub_list_female), delimiter = ',')
"""


r"""
efc_male = pd.read_csv(r"D:\Shraddha\FC_matrices_separated_male_female\S200_male.csv", header = None).values ##has the efc matrices of all males column wise
efc_female = pd.read_csv(r"D:\Shraddha\FC_matrices_separated_male_female\S200_female.csv", header = None).values ##has the efc matrices of all females column wise

sub_list_male = pd.read_csv(r"D:\Shraddha\Data\sub_list_male.csv", header = None).iloc[:, 0].values
sub_list_female = pd.read_csv(r"D:\Shraddha\Data\sub_list_female.csv", header = None).iloc[:, 0].values

for l in list([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):

    mean_efc_male = np.zeros([efc_male.shape[0], 100]) #for column wise storing of averaged efc
    mean_efc_female = np.zeros([efc_female.shape[0], 100])

    rand_sublist_male = np.zeros([l, 100]) #for column wise storing of averaged efc
    rand_sublist_female = np.zeros([l, 100])

    for k in range(100): #we sample l random subjects 100 times. That is l different subjects for 100 times. 

        rand_list_male = []
        rand_list_female = []
        while len(rand_list_male) <= l-1: #generate a list of random numbers between 0, 127 for males
            rand_num1 = random.randint(0, 127)
            if(rand_num1 in rand_list_male):
                continue
            else:
                rand_list_male.append(rand_num1)

        while len(rand_list_female) <= l-1: #generate a list of random numbers between 0, 143 for females
            rand_num2 = random.randint(0, 143)
            if(rand_num2 in rand_list_female):
                continue
            else:
                rand_list_female.append(rand_num2)

        
        summ1 = np.zeros(efc_male.shape[0])
        summ2 = np.zeros(efc_female.shape[0])
        for i in range(len(rand_list_male)):
            summ1 = summ1 + np.arctanh(efc_male[:, rand_list_male[i]]) #applyimg fisher z transform before summing it up

        for j in range(len(rand_list_female)):
            summ2 = summ2 + np.arctanh(efc_female[:, rand_list_female[j]]) #applying fisher z transform
            
        mean_efc_male[:, k] = np.tanh(summ1/len(rand_list_male)) #inverse fisher z 
        mean_efc_female[:, k] = np.tanh(summ2/len(rand_list_female)) #inverse fisher z

        rand_sublist_male[:, k] = sub_list_male[rand_list_male] #storing the list of l random sujects (column wise)
        rand_sublist_female[:, k] = sub_list_female[rand_list_female]

        #print(rand_list_male)
    #print(len(rand_list_female))

    #print(rand_sublist_male[:, 0])
    #print(rand_sublist_male)
    #print("hello")
    #print(rand_sublist_male[8, :])

    #print(len(rand_sublist_male))
    #print(len(rand_sublist_female))
    

    np.savetxt(r"D:\Shraddha\FC_matrices_subsampled\S200\S200_sets_of_" + str(l) + "_male.csv", mean_efc_male, delimiter = ',')
    np.savetxt(r"D:\Shraddha\FC_matrices_subsampled\S200\S200_sets_of_" + str(l) + "_female.csv", mean_efc_female, delimiter = ',')

    np.savetxt(r"D:\Shraddha\FC_matrices_subsampled\S200\sub_list_male_setsof_" + str(l) + ".csv", rand_sublist_male, delimiter = ',')
    np.savetxt(r"D:\Shraddha\FC_matrices_subsampled\S200\sub_list_female_setsof_" + str(l) + ".csv", rand_sublist_female, delimiter = ',')
    
"""

    