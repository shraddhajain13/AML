import numpy as np
import pandas as pd
import scipy
import glob
from scipy import stats
import matplotlib.pyplot as plt, mpld3
import collections
import ntpath
import os
import pandas as pd
import math
import antropy as ant
from multi_linear_reg import corr_sfc_efc_list, corr_sfc_esc_list, corr_eFC_eSC_list, corr_eFC_ePL_list, corr_eSC_ePL_list, gender_list_filtered, brain_size_list
import statistics
import pickle 
import plotly 
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

x = np.linspace(-1,1,21)
r"""
sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))
path = r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\Simulated_BOLD\sim_bold_int_1point5"
def entr_fc():
    vectors_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_sim_int_1point5.csv", header = None).values
    entr = np.zeros(272)
    stdn = np.zeros(272)
    for i in range(272):
        counts = plt.hist(vectors_fc[:, i], bins = x)[0]
        entr[i] = scipy.stats.entropy(counts) #shannon entropy of FC
        stdn[i] = np.std(abs(vectors_fc[:,i]))
    return entr, stdn
"""
def entr_sc():
    vectors_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_retry.csv", header = None).values
    ent_sc = np.zeros(272)
    for i in range(272):
        c = plt.hist(vectors_sc[:,i], bins = x)[0]
        ent_sc[i] = scipy.stats.entropy(c) #shannon entropy of SC
    return ent_sc
r"""
def entr_bold(): #for sample entropy
    samp_entr = np.zeros(272)
    for i in range(len(sub_num_list_old)):
        subject_number = str((int)(sub_num_list_old[i]))
        for filename in glob.glob(os.path.join(path, 'sim_bold_1point5_' + subject_number + '.csv')):
            print(subject_number)
            data = pd.read_csv(filename, header = None).values
            #e = 0
            se = 0
            for j in range(100):
                se = se + ant.sample_entropy(data[:,j])
            samp_entr[i] = se/100
    return samp_entr

def synchron(): #synchronicity
    vectors_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_sim_int_1point5.csv", header = None).values
    sync = np.zeros(272)
    for i in range(272):
        sync[i] = np.mean(abs(vectors_fc[i]))
    return sync

entropy_fc, stdn = entr_fc()
sample_entropy_bold = entr_bold()
synchronicity = synchron()
all_entropy = np.zeros([272, 5])
all_entropy[:,0] = np.array(sub_num_list_old)
all_entropy[:,1] = entropy_fc
all_entropy[:,2] = sample_entropy_bold
all_entropy[:,3] = synchronicity
all_entropy[:,4] = stdn

np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sync_sim_int_1point5.csv',all_entropy, delimiter=",") 
"""
#entropy_sc = entr_sc() #shannon entropy of SC
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sc.csv',entropy_sc, delimiter=",")

entropies = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sync.csv", header = None).values
entropy_fc = entropies[:,1] #shannon entropy of FC
entropy_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy_sc.csv", header = None).values 
sample_entropy_bold = entropies[:,2] #sample entropy of BOLD
synchroni = entropies[:,3] #synchronicity
stdn = entropies[:,4] #standard deviation
s = np.linspace(1,272,272)

r"""
sim_corr = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_array_zscored.csv", header = None).values
efc_esc = sim_corr[:, 1]
efc_epl = sim_corr[:, 2]
esc_epl = sim_corr[:, 3]

entropy_fc_male = []
entropy_fc_female = []
entropy_sc_male = []
entropy_sc_female = []
sampent_male = []
sampent_female = []
brain_size_male = []
brain_size_female = []
#def gender_seg():
for i in range(len(gender_list_filtered)):
    if (gender_list_filtered[i] == 'M'):
        entropy_fc_male.append(entropy_fc[i])
        entropy_sc_male.append(entropy_sc[i])
        sampent_male.append(sample_entropy_bold[i])
        brain_size_male.append(brain_size_list[i])
    if(gender_list_filtered[i] == 'F'):
        entropy_fc_female.append(entropy_fc[i])
        entropy_sc_female.append(entropy_sc[i])
        sampent_female.append(sample_entropy_bold[i])
        brain_size_female.append(brain_size_list[i])
"""
dataa = {'shan_fc':entropy_fc, 'samp_ent':sample_entropy_bold, 'brain_size':brain_size_list, 'gen':gender_list_filtered}
df = pd.DataFrame(data = dataa)
fig = px.scatter_3d(df, x='brain_size', y='shan_fc', z='samp_ent',
              color='gen')
fig.show()
fig.write_html("3Dfig.html")
    

#sim_1_corr = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\sim_1_corr_array_272.csv", header = None).values

#plt.plot(entropy_sc_male, entropy_fc_male, '.', label = 'Male')
#plt.plot(entropy_sc_female, entropy_fc_female, '.', label = 'Female')
#plt.xlabel('Shannon entropy of SC')
#plt.ylabel('Shannon entropy of FC')
#fig = plt.figure()
#ax = Axes3D(fig)
#fig = plt.figure()
#ax.scatter(brain_size_male, entropy_fc_male, sampent_male, label = 'Male')
#ax.scatter(brain_size_female, entropy_fc_female, sampent_female, label = 'Female')
#ax.set_xlabel("Brain size")
#ax.set_ylabel("Shannon entropy of FC")
#ax.set_zlabel('Sample entropy of BOLD')
#plt.title('Simulated BOLD with noise intensity = 1 * rand(0,1)')
#plt.savefig('3d_plot')
#plotly.offline.plot(fig, filename='file.html')
#dummy = mpld3.fig_to_html(fig)
#mpld3.save_html(fig, '3d_plot.html')
#pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
#print(dummy)
#fig1 = px.scatter_3d(x = [brain_size_male, brain_size_female], y = [entropy_fc_male, entropy_fc_female], z = [sampent_male, sampent_female])
#fig1.write_html("fig1.html")
#fig2 = px.scatter_3d(brain_size_male, entropy_fc_male, sampent_male)
#px.scatter_3d() 
#plt.legend() 
fig1.show()

#figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
#figx.show()
#plt.show()

r"""
corr_entrfc_sync = stats.pearsonr(entropy_fc, synchroni)[0]
corr_entrfc_sampen = stats.pearsonr(entropy_fc, sample_entropy_bold)[0]
corr_sampen_sync = stats.pearsonr(synchroni, sample_entropy_bold)[0]
print("corr(shannon entropy FC, Synchronicity)", corr_entrfc_sync)
print("corr(shannon entropy FC, Sample entropy BOLD)", corr_entrfc_sampen)
print("corr(Synchronicity, Sample entropy BOLD)", corr_sampen_sync)
#print("corr(sample_entropy, std)", stats.pearsonr(stdn, sample_entropy_bold)[0])
#print("corr(shannon_entropy, std)", stats.pearsonr(stdn, entropy_fc)[0])
#plt.plot(entropy_fc, stdn, '.')
#plt.xlabel('Shannon entropy of FC')
#plt.ylabel('Standard deviation of |FC|')
#plt.show()
"""
r"""
male_entropy = []
female_entropy = []
for i in range(272):
    if(gender_list_filtered[i] == 'M'):
        male_entropy.append(sample_entropy_bold[i])
    if(gender_list_filtered[i] == 'F'):
        female_entropy.append(sample_entropy_bold[i])

t_value, p_value = scipy.stats.ttest_ind(male_entropy, female_entropy)
print("t value = ", t_value)
print("p value = ", p_value)
"""
r"""
#corr_entr_sfc_efc = stats.pearsonr(entropy_fc, corr_sfc_efc)[0]
#corr_entr_sfc_esc = stats.pearsonr(entropy_fc, corr_sfc_esc_list)[0]
corr_entr_efc_esc = stats.pearsonr(sample_entropy_bold, efc_esc)[0]
corr_entr_efc_epl = stats.pearsonr(sample_entropy_bold, efc_epl)[0]
corr_entr_esc_epl = stats.pearsonr(sample_entropy_bold, esc_epl)[0]
#print("corr(shannon entropy, corr(sFC, eFC))", corr_entr_sfc_efc)
#print("corr(shannon entropy, corr(sFC, eSC))", corr_entr_sfc_esc)
print("corr(sample entropy BOLD, corr(eFC', eSC))", corr_entr_efc_esc)
print("corr(sample entropy BOLD, corr(eFC', ePL))", corr_entr_efc_epl)
print("corr(sample entropy BOLD, corr(eSC, ePL))", corr_entr_esc_epl)
#print(stats.pearsonr(sample_entropy_bold, entropy_fc)[0])
"""
r"""
data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\entropy.csv", header = None).values
e_fc = data[:,1]
samp = data[:,2]
#e_fc_old = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\old_entr_fc.csv", header = None).values
#print(e_fc_old[:,0])
print(stats.pearsonr(samp, e_fc)[0])
"""
