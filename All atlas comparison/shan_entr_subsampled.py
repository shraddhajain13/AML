import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
from scipy import stats

atlas  = "S200"
gender = "male"

path = "D:\\Shraddha\\FC_matrices_subsampled\\" + atlas
sets = np.linspace(10, 100, 10) #creating a list of sets of subjects averaged over

x = np.linspace(-1, 1, 21) #patition into 20 bins from -1 to 1

for i in range(len(sets)):

    data = pd.read_csv(os.path.join(path, atlas + "_sets_of_" + str(int(sets[i])) + "_" + gender +".csv"), header = None).values
    #print(data.shape[1])
    print(sets[i])
    shan_entr_efc = []
    for j in range(data.shape[1]):
        print(j)
        transformed_data = np.arctanh(data[:, j]) #fisher z transforming the data
        counts = plt.hist(transformed_data, bins = x)[0] #counting the number of points in each bin defined by x
        shan_entr_efc.append(scipy.stats.entropy(counts))

    np.savetxt("D:\\Shraddha\\shan_entr_subsampled\\" + atlas + "\\" + gender + "\\shan_entr_efc_sets_of_" + str(int(sets[i])) + ".csv", shan_entr_efc, delimiter = ',')
    