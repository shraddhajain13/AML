from audioop import avg
import numpy as np
import pandas as pd

# this program is for averaging SC and PL matrices across subjects. We exclude the elements that are 0 in both SC and PL matroces, because especially in PL, it gives a wrong estimate for length. 
# that is, the elements that are 0 in the PL matrix are 0 because they are disconnected (0 in SC), this does not mean that the length between the 2 regions is 0

atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45']
end_name_for_sc = ['SC_all', 'SC_all_Sch200', 'SC_all_Sch400', 'SC_all_Sch600', 'SC_all_Shen79', 'SC_all_Shen156', 'SC_all_Shen232', 'SC_all_HO', 'SC_HO_25th', 'SC_HO_35th', 'SC_HO_45th']
sub_num_list_old = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))


num_parcels = [100, 200, 400, 600, 79, 156, 232, 96, 96, 96, 95] #list containing the number of parcels for each atlas

## this function is for averaging the eSC matrices across subjects
def avg_sc(): #we check each element that is non zero and then avg it across subjects
    
    for i in range(len(atlas)): #looping over each atlas
        print(atlas[i])

        #creating a 3D array storing the eSC matrices of all subjects for a given atlas
        sc_3d = np.zeros([num_parcels[i], num_parcels[i], len(sub_num_list_old)])
        
        #ceating an empty 2D array to store the final average
        avg_esc = np.zeros([num_parcels[i], num_parcels[i]])

        for j, sub in enumerate(sub_num_list_old): #looping over subjects to fill up the matrix

            sub = sub = str((int)(sub))
            
            #filling up the matrix; j corresponds to the index of the subject. Average will be across all js
            sc_3d[:, :, j] = pd.read_csv("D:\\Shraddha\\SC_PL_matrices_all_atlas\\full_matrices\\eSC_full\\" + atlas[i]  + "\\esc_" + sub + ".csv", header = None).values

        #looping over each edge of sc_3d matrix; for each edge, we extract the value of this edge for all subjects.
        print("j lopp completed")
        for k in range(num_parcels[i]):
            for m in range(num_parcels[i]):

                each_edge_all_subjects = sc_3d[k, m, :] #etracting each edge for all subjects as a 1D array at once

                each_edge_all_subjects = each_edge_all_subjects[each_edge_all_subjects != 0] #eleminating all the zeros

                if(each_edge_all_subjects.shape[0] != 0): #only taking the mean if there is anything in the list
                    avg_esc[k, m] = np.mean(each_edge_all_subjects) #taking the mean

                #if an edge is zero for all subjects, then the above list 'each_edge_all_subjects' will be empty. So in the avg_esc matrix, that edge will be stored as zero
        
        print(avg_esc.shape)
        np.savetxt("D:\\Shraddha\\SC_PL_matrices_all_atlas_averaged\\full_averaged_matrices\\eSC_full_averaged\\esc_" + atlas[i] + ".csv", avg_esc, delimiter = ',')  

            
def avg_pl(): #we check each element that is non zero and then avg it across subjects
    
    for i in range(len(atlas)): #looping over each atlas
        print(atlas[i])

        #creating a 3D array storing the ePL matrices of all subjects for a given atlas
        pl_3d = np.zeros([num_parcels[i], num_parcels[i], len(sub_num_list_old)])
        
        #ceating an empty 2D array to store the final average
        avg_epl = np.zeros([num_parcels[i], num_parcels[i]])

        for j, sub in enumerate(sub_num_list_old): #looping over subjects to fill up the matrix

            sub = sub = str((int)(sub))
            
            #filling up the matrix; j corresponds to the index of the subject. Average will be across all js
            pl_3d[:, :, j] = pd.read_csv("D:\\Shraddha\\SC_PL_matrices_all_atlas\\full_matrices\\ePL_full\\" + atlas[i]  + "\\epl_" + sub + ".csv", header = None).values

        #looping over each edge of pl_3d matrix; for each edge, we extract the value of this edge for all subjects.
        print("j loop completed")
        for k in range(num_parcels[i]):
            for m in range(num_parcels[i]):

                each_edge_all_subjects = pl_3d[k, m, :] #etracting each edge for all subjects as a 1D array at once

                each_edge_all_subjects = each_edge_all_subjects[each_edge_all_subjects != 0] #eleminating all the zeros

                if(each_edge_all_subjects.shape[0] != 0): #only taking the mean if there is anything in the list
                    avg_epl[k, m] = np.mean(each_edge_all_subjects) #taking the mean

                #if an edge is zero for all subjects, then the above list 'each_edge_all_subjects' will be empty. So in the avg_esc matrix, that edge will be stored as zero
        
        print(avg_epl.shape)
        np.savetxt("D:\\Shraddha\\SC_PL_matrices_all_atlas_averaged\\full_averaged_matrices\\ePL_full_averaged\\epl_" + atlas[i] + ".csv", avg_epl, delimiter = ',')  

avg_pl()
            

























