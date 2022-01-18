import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


sets = np.linspace(10, 100, 10) #creating a list of sets of subjects averaged over
l = len(sets)
gender = "female"
atlas = "S200"

r"""
#D:\Shraddha\Synch_subsampled\S200

file = "D:\\Shraddha\\shan_entr_subsampled\\" + atlas + "\\" + gender + "\\shan_entr_efc_sets_of_"
data = []
for i in range(len(sets)):
    data.append(pd.read_csv(file + str(int(sets[i]))+ ".csv", header = None).values[:, 0])

#print(np.array(data).shape)
np.savetxt("D:\\Shraddha\\Data\\subsample_shan_entr_" + atlas + "_"+ gender + "_all_sets.csv", (np.array(data).transpose()), delimiter = ',') #storing the shan etr of all sets together in one file column wise. so 100*10
#print(np.array(data).transpose())
"""

sets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data = pd.read_csv("D:\\Shraddha\\Data\\subsample_std_" + atlas + "_"+ gender + "_all_sets.csv", header = None).values


def set_box_color(bp, color): #setting color for box plots

    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))
box_plots = plt.boxplot(np.array(data), positions = np.array(range(l))*2)

set_box_color(box_plots, 'blue') 

plt.xticks(range(0, (l * 2), 2), sets, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Number of sets sub-sampled', fontsize = 20)
plt.ylabel('Std dev of eFC', fontsize = 20)
plt.tight_layout()
plt.show()


    


