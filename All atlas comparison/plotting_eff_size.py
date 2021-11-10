import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath
#import statsmodels.api as sm
from statsmodels.stats import multitest


r"""
#sub_num_list = np.loadtxt("C:\\Users\\shrad\\OneDrive\\Desktop\\Juelich\\Internship\\Data\\List_23_28_54_49_118.txt", usecols=(0))
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0', 'HO25', 'HO35', 'HO45'] 


before_reg = pd.read_csv(r"E:\Shraddha\Data\eff_size_p_val_corr_sfc_esc_lc_br.csv", header = None).values
eff_size_br = before_reg[:, 0]
p_val_br = multitest.fdrcorrection(before_reg[:, 1])[1]


after_reg_brainsize = pd.read_csv(r"E:\Shraddha\Data\eff_size_p_val_corr_sfc_esc_lc_arb.csv", header = None).values
eff_size_arb = after_reg_brainsize[:, 0]
p_val_arb = multitest.fdrcorrection(after_reg_brainsize[:, 1])[1]


after_reg_brainsize_empcorr = pd.read_csv(r"E:\Shraddha\Data\eff_size_p_val_corr_sfc_esc_lc_arbc.csv", header = None).values
eff_size_arbc = after_reg_brainsize_empcorr[:, 0]
p_val_arbc = multitest.fdrcorrection(after_reg_brainsize_empcorr[:, 1])[1]


d = {'Atlas': atlas, 'Effect size before reg': eff_size_br, 'P val before reg': p_val_br, 'Effect size after reg brain size': eff_size_arb, 'P val after regression brain size':p_val_arb, 'Effect size after reg brain size and emp corr': eff_size_arbc, 'P val after reg brain size and emp corr': p_val_arbc}
df = pd.DataFrame(data = d)

df.to_csv(r"E:\Shraddha\Data\eff_size_p_val_all_combined_corr_sfc_esc_lc_model.csv", index = False)
"""

r"""
#####.....This is for empirical correlations.....######
 
before_reg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_br.csv", header = None).values
eff_size_br = before_reg[:, 0]
p_val_br = multitest.fdrcorrection(before_reg[:, 1])[1]


after_reg_brainsize = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_emp_arb.csv", header = None).values
eff_size_arb = after_reg_brainsize[:, 0]
p_val_arb = multitest.fdrcorrection(after_reg_brainsize[:, 1])[1]

d = {'Atlas': atlas, 'Effect size before reg': eff_size_br, 'P val before reg': p_val_br, 'Effect size after reg brain size': eff_size_arb, 'P val after regression brain size':p_val_arb}
df = pd.DataFrame(data = d)
df.to_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\emp_eff_size_p_val.csv", index = False)

#########################################################


#####.....This is for optimal coupling strength for corr(sFC, eFC).....######
 
before_reg = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_coup_lc_br.csv", header = None).values
eff_size_br = before_reg[:, 0]
p_val_br = multitest.fdrcorrection(before_reg[:, 1])[1]


after_reg_brainsize = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\eff_size_p_val_coup_lc_arb.csv", header = None).values
eff_size_arb = after_reg_brainsize[:, 0]
p_val_arb = multitest.fdrcorrection(after_reg_brainsize[:, 1])[1]

d = {'Atlas': atlas, 'Effect size before reg': eff_size_br, 'P val before reg': p_val_br, 'Effect size after reg brain size': eff_size_arb, 'P val after regression brain size':p_val_arb}
df = pd.DataFrame(data = d)
df.to_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\coup_lc_eff_size_p_val.csv", index = False)

#########################################################


"""

data = pd.read_csv(r"E:\Shraddha\Data\eff_size_p_val_all_combined_corr_sfc_esc_lc_model.csv").values
atlas = ['S100', 'S200', 'S400', 'S600', 'Shen79', 'Shen156', 'Shen232', 'HO0%', 'HO25%', 'HO35%', 'HO45%'] 
plt.rcParams['font.size'] = '20'
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (19, 10))
ax1.plot(np.array(atlas), data[:, 1], label = 'Before regression', marker = '.', markersize = 10)
ax1.plot(np.array(atlas), data[:, 3], label = 'After regression of brain size', marker = '.', markersize = 10)
ax1.plot(np.array(atlas), data[:, 5], label = 'After regression of brain size and corr(eFC, eSC)', marker = '.', markersize = 10)
#ax1.set_ylim(0, 1)
ax1.yaxis.grid()
ax1.set_title('Hedges g vs Atlas - Limit Cycle model')
ax1.set_ylabel('Hedges g', fontsize = 22)

ax2.plot(np.array(atlas), data[:, 2], label = 'Before regression', marker = '.', markersize = 10)
ax2.plot(np.array(atlas), data[:, 4], label = 'After regression of brain size', marker = '.', markersize = 10)
ax2.plot(np.array(atlas), data[:, 6], label = 'After regression of brain size and corr(eFC, eSC)', marker = '.', markersize = 10)
ax2.set_title('P value (FDR corrected) vs Atlas')
ax2.set_ylabel('P value', fontsize = 22)
ax2.set_xlabel('Atlas', fontsize = 22)
ax2.axhline(y = 0.05, color = 'r', linestyle = '--')#, label = 'threshold $/alpha$')
ax2.yaxis.grid()
plt.xticks(rotation = 45)
#plt.legend()
ax1.legend(bbox_to_anchor=(1, 1.0), loc='upper left')
plt.tight_layout()
#plt.savefig(r"C:\Users\shrad\Desktop\phase_eff_size_p_val_temp.png", dpi = 300)
plt.show()
