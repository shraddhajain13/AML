import numpy as np
import pandas as pd
import scipy
from scipy import stats
from maxcorr_gender import max_corr_list_fc, max_corr_list_sc, gender_list, subject_number_list_sim
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pingouin




vectors_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_zscored.csv", header = None).values
vectors_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_retry.csv", header = None).values
vectors_pl = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_retry.csv", header = None).values

sub_num_list_old = np.loadtxt(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\List_23_28_54_49_118.txt",usecols=(0))  ## this is the list of 272 subjects that we want to investigate

def rearr(list_1):
    list_2 = []
    for i in range(272):
        ind = np.where(subject_number_list_sim == sub_num_list_old[i])[0][0]
        #print("i = ", i)
        #print("index = ", ind)
        #print(list_1[ind])
        list_2.append(list_1[ind])
    return(list_2)

def cohens_D(list1, list2): #function for finding the Cohen's D
    var1 = np.var(list1)
    var2 = np.var(list2)
    n1 = len(list1)
    n2 = len(list2)
    x1 = mean(list1)
    x2 = mean(list2)
    pooled_stdev = math.sqrt((((n1-1)*var1) + ((n2-1)*var2))/(n1+n2-1))
    D = (x1-x2)/pooled_stdev
    return D
#print(max_corr_list_fc)    
#print(rearr(max_corr_list_fc))
corr_sfc_efc_list = rearr(max_corr_list_fc)
corr_sfc_esc_list = rearr(max_corr_list_sc)
corr_eFC_eSC_list = []
corr_eFC_ePL_list = []
corr_eSC_ePL_list = []
#print(len(corr))

for i in range(272):
    corr_eFC_eSC = stats.pearsonr(vectors_fc[:,i], vectors_sc[:,i])[0]
    corr_eFC_ePL = stats.pearsonr(vectors_fc[:,i], vectors_pl[:,i])[0]
    corr_eSC_ePL = stats.pearsonr(vectors_sc[:,i], vectors_pl[:,i])[0]
    corr_eFC_eSC_list.append(corr_eFC_eSC)
    corr_eFC_ePL_list.append(corr_eFC_ePL)
    corr_eSC_ePL_list.append(corr_eSC_ePL)
    

corr_array = np.zeros([272, 4])
corr_array[:,0] = np.array(sub_num_list_old)
corr_array[:,1] = corr_eFC_eSC_list 
corr_array[:,2] = corr_eFC_ePL_list 
corr_array[:,3] = corr_eSC_ePL_list
np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\corr_array_zscored.csv', corr_array, delimiter=",")
#print(corr_array) 


phen_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv") ##for phenotypical data
sub_num_list_phen = phen_data.iloc[:,0].values
brain_size_phen = phen_data.iloc[:,192].values
brain_size_list = []
gender_list_filtered = []


for i in range(272):
    index = np.where(sub_num_list_phen == sub_num_list_old[i])[0][0]
    brain_size_list.append(brain_size_phen[index])
    gender_list_filtered.append(gender_list[index])


#print(sub_num_list_old)
#print(gender_list_filtered)

X = np.zeros([272,2]) # column #1 is the corr_eFC_eSC, column #2 is the brain size
X[:,0] = np.array(corr_eSC_ePL_list)
X[:,1] = np.array(brain_size_list) 
#print(stats.pearsonr(X[:,0], X[:,1])[0])
#x = np.array(corr_eFC_eSC_list).reshape((-1, 1))
Y = np.array(corr_sfc_esc_list) # column #2 is the corr_sFC_eFC 
reg = LinearRegression().fit(X, Y)
#print(reg.coef_)
Y_hat = reg.predict(X)
print("coefficient of determination = ", reg.score(X, Y))
residue = Y - Y_hat #list of 272 residues
male_res_list = []
female_res_list = []
male_brain_list = []
female_brain_list = []
male_list_fc = []
male_list_sc = []
female_list_fc = []
female_list_sc = []
#print(residue)
for i in range(272):
    if(gender_list_filtered[i] == 'M'):
        #male_list_fc.append(corr_sfc_efc_list[i])
        male_list_sc.append(corr_sfc_esc_list[i])
        male_res_list.append(residue[i])
        #male_brain_list.append(brain_size_list[i])
    if(gender_list_filtered[i] == 'F'):
        #female_list_fc.append(corr_sfc_efc_list[i])
        female_list_sc.append(corr_sfc_esc_list[i])
        female_res_list.append(residue[i])
        #female_brain_list.append(brain_size_list[i])

#print(male_res_list)
#plt.plot(Y, residue, '.')
#plt.title('Y = corr(sFC, eFC) X = corr(eFC, eSC)')
#plt.ylabel('Residue')
#plt.xlabel('Original Y')
#plt.show()
#print(stats.pearsonr(residue, Y)[0])

t_value, p_value = scipy.stats.ttest_ind(male_res_list, female_res_list)
#t_value, p_value = scipy.stats.ranksums(male_res_list, female_res_list)
print("t value = ", t_value)
print("p value = ", p_value)
eff_size = pingouin.compute_effsize(male_res_list, female_res_list)
print("Effect size = ", eff_size)


