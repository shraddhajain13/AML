import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


threshold = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
l = len(threshold)
r"""
#D:\Shraddha\Synch_subsampled\S200
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
gender_list = pheno_data[:, 3]

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list_old[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

gender_list_filtered = rearr(gender_list)


def categorise_male_female(x): # function to split the list into M and F ; x is the list that has to be split into M and F
    list1 = [] #for males
    list2 = [] #for females 
    for i in range(272):
        if(gender_list_filtered[i] == 'M'):
            list1.append(x[i])
        if(gender_list_filtered[i] == 'F'):
            list2.append(x[i])
    return list1, list2

output_male = []
output_female = []
for i in range(len(threshold)):
    data = pd.read_csv("D:\\Shraddha\\Thresholding\\std_dev\\std_dev" + str(threshold[i])+ ".csv").values[:, 1]
    data_male, data_female = categorise_male_female(data)
    output_male.append(data_male)
    output_female.append(data_female)
    
np.savetxt("D:\\Shraddha\\Thresholding\\std_dev\\std_dev_all_threshold_male.csv", (np.array(output_male).transpose()), delimiter = ',') 
np.savetxt("D:\\Shraddha\\Thresholding\\std_dev\\std_dev_all_threshold_female.csv", (np.array(output_female).transpose()), delimiter = ',') 

#print(np.array(output_female).transpose().shape)
"""
def set_box_color(bp, color): #setting color for box plots
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


data_male = pd.read_csv("D:\\Shraddha\\Thresholding\\std_dev\\std_dev_all_threshold_male.csv", header = None).values
data_female = pd.read_csv("D:\\Shraddha\\Thresholding\\std_dev\\std_dev_all_threshold_female.csv", header = None).values


plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))

male_plots = plt.boxplot(data_male, positions = np.array(range(l))*2 - 0.3)
female_plots = plt.boxplot(data_female, positions = np.array(range(l))*2 + 0.3)

set_box_color(male_plots, 'blue') 
set_box_color(female_plots, 'red')

plt.plot([], c='blue', label='Male')
plt.plot([], c='red', label='Female')
plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #only for the purpose of x axis

plt.xticks(range(0, (l * 2), 2), ticks, rotation = 0)
plt.xlim(-2, (l*2))
plt.xlabel('Threshold', fontsize = 20)
plt.ylabel('Std dev of eFC', fontsize = 20)
plt.tight_layout()
plt.show()