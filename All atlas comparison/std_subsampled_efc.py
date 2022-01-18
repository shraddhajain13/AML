#This program is to calculate the synchronicity of the subsampled eFC (100 per set) for sets of 10, 20, 30....,100 subjects

import numpy as np
import pandas as pd
import os

sets = np.linspace(10, 100, 10) #creating a list of sets of subjects averaged over

gender = "female"
atlas = "S200"
path = "D:\\Shraddha\\FC_matrices_subsampled\\" + atlas

for i in range(len(sets)):

    data = pd.read_csv(os.path.join(path, atlas + "_sets_of_" + str(int(sets[i])) + "_" + gender +".csv"), header = None).values
    print(sets[i])
    std_list = []
    for j in range(data.shape[1]):
        print(j)
        transformed_data = np.arctanh(data[:, j]) #fisher z transform
        std_list.append(np.tanh(np.std(abs(transformed_data)))) #calculating the std dev and then back transform
    
    np.savetxt("D:\\Shraddha\\std_subsampled\\" + atlas + "\\" + gender + "\\std_efc_sets_of_" + str(int(sets[i])) + ".csv", std_list, delimiter = ',')


    


