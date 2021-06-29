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
import antropy as ant
import statistics
#import colorednoise as cn
import random


sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Concatenated"


def first_bold(subject_number):
    for filename in glob.glob(os.path.join(path, 'conc' + subject_number + '.csv')):
        #print(subject_number)
        data = pd.read_csv(filename, header = None).values
    return data[:,0]

def add_noise(first_column): #Noise is added at every time step, and for eavery parcellation. 
    #np.random.seed(i)
    intensity = 1.5
    x = np.zeros(4800)
    for i in range(4800):
        noise = np.random.normal(size = 1) #noise
        x[i] = first_column[i] + (intensity * noise)
    return x

for i in range(len(sub_num_list_old)):
    subject_number = str((int)(sub_num_list_old[i]))
    print(subject_number)
    first_column = first_bold(subject_number)
    bold = np.zeros([4800, 100])
    for i in range(100):
        bold[:, i] = add_noise(first_column)
    np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Simulated_BOLD\sim_bold_int_1point5\sim_bold_1point5_'+subject_number+'.csv', bold , delimiter=",") 


r"""
beta = 1 # the exponent
samples = 4800 # number of samples to generate
perm_ent = np.zeros(100)
samp_ent = np.zeros(100)
shannon_ent = np.zeros(100)
for i in range(100):
    
    #np.random.seed(i)
    x = cn.powerlaw_psd_gaussian(beta, samples)
    #x = np.random.normal(size = 3000)
    #print(i)
    #print(x)
    perm_ent[i] = ant.perm_entropy(x)
    samp_ent[i] = ant.sample_entropy(x)
    counts = plt.hist(x, bins = 20)[0]
    shannon_ent[i] = scipy.stats.entropy(counts)

corr1 = stats.pearsonr(perm_ent, samp_ent)[0]
corr2 = stats.pearsonr(samp_ent, shannon_ent)[0]
corr3 = stats.pearsonr(shannon_ent, perm_ent)[0]

print(corr1)
print(corr2)
print(corr3)

x = np.random.normal(size = 4800)
#print(x)
print(np.mean(x))
print(np.std(x))
"""