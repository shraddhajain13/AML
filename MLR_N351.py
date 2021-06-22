import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pingouin

from maxcorr_gender_351_subs import sub_num_list_351_ordered, max_corr_list_fc_351, max_corr_list_sc_351


vectors_fc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_fc_concatenated_N351.csv", header = None).values
vectors_sc = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_sc_N351.csv", header = None).values
vectors_pl = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\output_pl_N351.csv", header = None).values


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


corr_eFC_eSC_list = []
corr_eFC_ePL_list = []
corr_eSC_ePL_list = []


for i in range(351):
    corr_eFC_eSC = stats.pearsonr(vectors_fc[:,i], vectors_sc[:,i])[0]
    corr_eFC_ePL = stats.pearsonr(vectors_fc[:,i], vectors_pl[:,i])[0]
    corr_eSC_ePL = stats.pearsonr(vectors_sc[:,i], vectors_pl[:,i])[0]
    corr_eFC_eSC_list.append(corr_eFC_eSC)
    corr_eFC_ePL_list.append(corr_eFC_ePL)
    corr_eSC_ePL_list.append(corr_eSC_ePL)
    

corr_array = np.zeros([351, 4])
corr_array[:,0] = np.array(sub_num_list_351_ordered)
corr_array[:,1] = corr_eFC_eSC_list 
corr_array[:,2] = corr_eFC_ePL_list 
corr_array[:,3] = corr_eSC_ePL_list
#np.savetxt(r'C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\All_three_corrs_array_N351.csv', corr_array, delimiter=",")
#print(corr_array) 



phen_data = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv") ##for phenotypical data
phen_data_rest = pd.read_csv(r"C:\Users\shrad\OneDrive\Desktop\Juelich\Internship\Data\RESTRICTED_shraddhajain13_3_3_2021_3_41_5.csv") ##restricted
sub_num_list_phen = phen_data.iloc[:,0].values
gender_list = phen_data.iloc[:,3].values
brain_size_phen = phen_data.iloc[:,192].values
age_phen = phen_data_rest.iloc[:,1].values
brain_size_list = []
gender_list_filtered = []
age_filtered = []
#print(age_phen)


for i in range(351):
    index = np.where(sub_num_list_phen == sub_num_list_351_ordered[i])[0][0]
    brain_size_list.append(brain_size_phen[index])
    gender_list_filtered.append(gender_list[index])
    age_filtered.append(age_phen[index])

#print("Subject Number = ", sub_num_list_351_ordered)
#print("Age = ", age_filtered)
#print("Brain Size = ", brain_size_list)
#print(sub_num_list_old)
#print(sub_num_list_old)
#print("Gender list = ", gender_list_filtered)

X = np.zeros([351,2]) # column #1 is the corr_eFC_eSC, column #2 is the brain size
X[:,0] = np.array(corr_eSC_ePL_list)
X[:,1] = np.array(brain_size_list) 
#X[:,2] = np.array(age_filtered)
#print(stats.pearsonr(X[:,0], X[:,1]))
#x = np.array(corr_eFC_eSC_list).reshape((-1, 1))
Y = np.array(max_corr_list_sc_351)  

corr_pear_before_reg, p_pear_before_reg = stats.pearsonr(Y, age_filtered)
print("Pearson Correlation before regression = ", corr_pear_before_reg)
print("p - pearson before regression = ", p_pear_before_reg)

reg = LinearRegression().fit(X, Y)
Y_hat = reg.predict(X)
residue = Y - Y_hat 

corr_pear_after_reg, p_pear_after_reg = stats.pearsonr(residue, age_filtered)
print("Pearson correlation after regression = ", corr_pear_after_reg)
print("p - pearson  after regression = ", p_pear_after_reg)

corr_spear_before_reg, p_spear_before_reg = stats.spearmanr(Y, age_filtered)
print("Spearman Correlation before regression = ", corr_spear_before_reg)
print("p - spearman before regression = ", p_spear_before_reg)


corr_spear_after_reg, p_spear_after_reg = stats.spearmanr(residue, age_filtered)
print("Spearman Correlation before regression = ", corr_spear_after_reg)
print("p - spearman before regression = ", p_spear_after_reg)

r"""
male_res_list = []
female_res_list = []
male_brain_list = []
female_brain_list = []
male_list_fc = []
male_list_sc = []
female_list_fc = []
female_list_sc = []
avg_male_list_sc = []
avg_female_list_sc = []
#print(residue)
for i in range(351):
    if(gender_list_filtered[i] == 'M'):
        male_res_list.append(residue[i])
    if(gender_list_filtered[i] == 'F'):
        female_res_list.append(residue[i])
        
t_value, p_value = scipy.stats.ttest_ind(male_res_list, female_res_list)
eff_size = pingouin.compute_effsize(male_res_list, female_res_list)

print("coefficient of determination = ", reg.score(X, Y))
print("t value = ", t_value)
print("p value = ", p_value)
print("Effect size = ", eff_size)
print("Correlation = ", stats.pearsonr(residue, Y)[0])
#print(male_res_list)
#plt.plot(Y, residue, '.')
#plt.title('Y = corr(sFC, eFC) X = corr(eFC, eSC)')
#plt.ylabel('Residue')
#plt.xlabel('Original Y')
#plt.show()
"""