import numpy as np
import pandas as pd
import scipy
import glob
from scipy import stats
import matplotlib.pyplot as plt
import collections
import ntpath
import os
import pandas as pd
import math
import statistics

sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
efc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_sim_int_2.csv", header = None).values
esc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_retry.csv", header = None).values
epl = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_retry.csv", header = None).values
sim_corr_efc_esc = np.zeros(272)
sim_corr_efc_epl = np.zeros(272)
sim_corr_esc_epl = np.zeros(272)
#print(esc[:,0].shape)

for i in range(272):
    sim_corr_efc_esc[i] = stats.pearsonr(efc[:,i], esc[:,i])[0]
    sim_corr_efc_epl[i] = stats.pearsonr(efc[:,i], epl[:,i])[0]
    sim_corr_esc_epl[i] = stats.pearsonr(esc[:,i], epl[:,i])[0]

sim_corr_array_272 = np.zeros([272,4])
sim_corr_array_272[:,0] = sub_num_list_old
sim_corr_array_272[:,1] = sim_corr_efc_esc
sim_corr_array_272[:,2] = sim_corr_efc_epl
sim_corr_array_272[:,3] = sim_corr_esc_epl

np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\sim_2_corr_array_272.csv', sim_corr_array_272 , delimiter=",")
