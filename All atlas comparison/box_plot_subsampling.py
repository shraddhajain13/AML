import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#This program is to calculate the synchronicity of the subsampled eFC (100 per set) for sets of 10, 20, 30....,100 subjects

import numpy as np
from numpy.core.getlimits import MachArLike
import pandas as pd
import os

sets = np.linspace(10, 100, 10) #creating a list of sets of subjects averaged over
l = len(sets)
gender = "female"
atlas = "Shen_79"

r"""
file = "D:\\Shraddha\\synch_subsampled\\" + atlas + "\\" + gender + "\\synch_efc_sets_of_"
data = []
for i in range(len(sets)):
    data.append(pd.read_csv(file + str(int(sets[i]))+ ".csv", header = None).values[:, 0])

np.savetxt("D:\\Shraddha\\Data\\subsample_synch_" + atlas + "_"+ gender + "_all_sets.csv", (np.array(data).transpose()), delimiter = ',')
#print(np.array(data).transpose())
"""


sets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data = pd.read_csv("D:\\Shraddha\\Data\\subsample_synch_" + atlas + "_"+ gender + "_all_sets.csv", header = None).values


def set_box_color(bp, color): #setting color for box plots

    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
plt.rcParams['font.size'] = '20'
plt.figure(figsize = (16, 8))
box_plots = plt.boxplot(np.array(data), positions = np.array(range(l))*2)

set_box_color(box_plots, 'red') 

plt.xticks(range(0, (l * 2), 2), sets, rotation = 45)
plt.xlim(-2, (l*2))
plt.xlabel('Number of sets sub-sampled', fontsize = 20)
plt.ylabel('Synchronicity of eFC', fontsize = 20)
plt.tight_layout()
plt.show()


    


