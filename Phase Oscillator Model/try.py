import numpy as np
import math
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import os
import ntpath

r"""
data = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase\A9F02Ext100Tri_CPU100Tri_CPU2100Tri_CPU3100Tri_CPU4100Tri_CPU5_bif_max", usecols = (29, 30, 31, 32, 33, 34))
data2 = np.loadtxt(r"C:\Users\shrad\Downloads\A9F0002Ext_GPU_GPU2_GPU3_GPU4_GPU5_bif_max", usecols = (3))
#print(data2)
#print(data[0, 0])
#print(data[0, 1])
#print(data[0, 2])
k = 0
#delay_fc_list = []
#delay_sc_list = []
#coup_str_fc_list = []
#coup_str_sc_list = []
corr_sfc_efc_list1 = []
corr_sfc_efc_list2 = []
#corr_sfc_esc_list = []

for i in range(272): #loop to extract the goodness of fit for each subject. It is the first row for each subject in the file of each parcellation.
    #delay_sfc_efc = data[k, 0]
    #coup_str_sfc_efc = data[k, 1]
    corr_sfc_efc1 = data[k, 2]
    corr_sfc_efc2 = data2[k]
    #delay_sfc_esc = data[k, 3]
    #coup_str_sfc_esc = data[k, 4]
    #corr_sfc_esc = data[k, 5]

    #delay_fc_list.append(delay_sfc_efc)
    #delay_sc_list.append(delay_sfc_esc)
    #coup_str_fc_list.append(coup_str_sfc_efc)
    #coup_str_sc_list.append(coup_str_sfc_esc)
    corr_sfc_efc_list1.append(corr_sfc_efc1)
    corr_sfc_efc_list2.append(corr_sfc_efc2)

    k = k + 25

#print(corr_sfc_efc_list1)
#print(corr_sfc_efc_list2)
#corr_sfc_efc_list1 = np.arctanh(corr_sfc_efc_list1) #fisher z transform
#corr_sfc_efc_list2 = np.arctanh(corr_sfc_efc_list2) #fisher z transform

#k_test1_D, p1 = scipy.stats.kstest(corr_sfc_efc_list1, 'norm')
#print(k_test1_D, p1)

t_value, p_value = scipy.stats.ranksums(corr_sfc_efc_list1, corr_sfc_efc_list2)
#print(t_value, p_value)
x = range(0, 24, 2)
print(x)
"""
r"""
data_a = np.array([[1,2,5], [5,7,2,2,5], [7,2,5]])
data_b = np.array([[6,4,2], [1,2,5,3,2], [2,3,5,1]])

ticks = ['A', 'B', 'C']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Apples')
plt.plot([], c='#2C7BB6', label='Oranges')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 8)
plt.tight_layout()
#plt.show()
#plt.savefig('boxcompare.png')
print(data_a.shape)
"""
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\00_Fitting_results_11Parcellations_Phase_LC\Phase"
parcel_list = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\list_of_parcellations.txt", dtype = str)
#print(parcel_list)
for filename in parcel_list:
    n = os.path.join(path, filename)
    print(n)
    data = np.loadtxt(n, usecols = (29, 30, 31, 32, 33, 34))
    print(data[:, 2])
    break
    #print(type(filename))