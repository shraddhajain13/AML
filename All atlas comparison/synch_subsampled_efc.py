#This program is to calculate the synchronicity of the subsampled eFC (100 per set) for sets of 10, 20, 30....,100 subjects

import numpy as np
import pandas as pd
import os

sets = np.linspace(10, 100, 10) #creating a list of sets of subjects averaged over

gender = "female"
atlas = "Shen_79"
path = "D:\\Shraddha\\FC_matrices_subsampled\\" + atlas

for i in range(len(sets)):

    data = pd.read_csv(os.path.join(path, atlas + "_sets_of_" + str(int(sets[i])) + "_" + gender +".csv"), header = None).values
    
    synch_list = []
    for j in range(data.shape[1]):
        synch_list.append(np.mean(abs(data[:, j])))
    
    np.savetxt("D:\\Shraddha\\Synch_subsampled\\" + atlas + "\\" + gender + "\\synch_efc_sets_of_" + str(int(sets[i])) + ".csv", synch_list, delimiter = ',')


    


