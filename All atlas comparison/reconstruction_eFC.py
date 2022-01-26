#this program recalculates eFC matrices using the upper triangular parts of all subjects for all atlases


import numpy as np
import pandas as pd

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))
num_parcels = [100, 200, 400, 600, 79, 156, 232, 96, 96, 96, 95] #list containing the number of parcels for each atlas

for i in range(len(atlas)):
    efc_all_subjects = pd.read_csv("D:\\Shraddha\\FC_matrices_all_atlas\\" + "efc_" + atlas[i] + ".csv", header = None).values
    
    print(atlas[i])

    for j, sub in enumerate(sub_num_list_old):

        sub = str((int)(sub))

        upp_tri_part = efc_all_subjects[:, j] #extracting the upper triangular part for a subject
        
        #creating an empty matrix to store the eFC of a subject. 
        efc = np.zeros([num_parcels[i], num_parcels[i]])

        n = 0 #index for the upp_tri_part list

        for k in range(num_parcels[i]):
            for m in range(k, num_parcels[i]):

                if(k==m):
                    efc[k, m] = 1 #diagonal elements are 1 in eFC matrix

                else:
                    efc[k, m] = upp_tri_part[n] #rearranging the upper triangular part into a matrix
                    efc[m, k] = efc[k, m] #because it is a symmetric matrix
                    n = n + 1
        
       
        np.savetxt("D:\\Shraddha\\FC_matrices_all_atlas\\full_matrices\\" + atlas[i] + "\\efc_" + sub + ".csv", efc, delimiter = ',')
        
    print(efc.shape)

