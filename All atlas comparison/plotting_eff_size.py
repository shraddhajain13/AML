import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath


#sub_num_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45'] 

r"""
before_reg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_lc_br.csv", header = None).values
eff_size_br = before_reg[:, 0]
p_val_br = before_reg[:, 1]


after_reg_brainsize = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_lc_arb.csv", header = None).values
eff_size_arb = after_reg_brainsize[:, 0]
p_val_arb = after_reg_brainsize[:, 1]


after_reg_brainsize_empcorr = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_lc_arbc.csv", header = None).values
eff_size_arbc = after_reg_brainsize_empcorr[:, 0]
p_val_arbc = after_reg_brainsize_empcorr[:, 1]

arr = np.zeros([11, 7])

d = {'Atlas': atlas, 'Effect size before reg': eff_size_br, 'P val before reg': p_val_br, 'Effect size after reg brain size': eff_size_arb, 'P val after regression brain size':p_val_arb, 'Effect size after reg brain size and emp corr': eff_size_arbc, 'P val after reg brain size and emp corr': p_val_arbc}
df = pd.DataFrame(data = d)
df.to_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\lc_model_eff_size_p_val.csv", index = False)

"""
r"""
#####.....This is for empirical correlations.....######
 
before_reg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_br.csv", header = None).values
eff_size_br = before_reg[:, 0]
p_val_br = before_reg[:, 1]


after_reg_brainsize = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_arb.csv", header = None).values
eff_size_arb = after_reg_brainsize[:, 0]
p_val_arb = after_reg_brainsize[:, 1]

d = {'Atlas': atlas, 'Effect size before reg': eff_size_br, 'P val before reg': p_val_br, 'Effect size after reg brain size': eff_size_arb, 'P val after regression brain size':p_val_arb}
df = pd.DataFrame(data = d)
df.to_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\emp_eff_size_p_val.csv", index = False)

#########################################################
"""



data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\emp_eff_size_p_val.csv").values
plt.plot(data[:, 0], data[:, 2], label = 'Before regression', marker = '.', markersize = 10)
plt.plot(data[:, 0], data[:, 4], label = 'After regression of brain size', marker = '.', markersize = 10)
#plt.plot(data[:, 0], data[:, 5], label = 'After regression of brain size and corr(eFC, eSC)', marker = '.', markersize = 10)
plt.title('Empirical correlations')
plt.ylabel('Significance - P value')
plt.xlabel('Atlas')
plt.legend()
plt.show()
