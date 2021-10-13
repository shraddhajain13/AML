import numpy as np
import pandas as pd

sub_num_list = np.loadtxt(r"D:\Shraddha\Data\List_23_28_54_49_118.txt", usecols=(0))
corr_sfc_efc_phase = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_phase.csv", header = None).values[:, 1]
corr_sfc_esc_phase = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_esc_all_atlas_phase.csv", header = None).values[:, 1]
corr_sfc_efc_lc = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_efc_all_atlas_lc.csv", header = None).values[:, 1]
corr_sfc_esc_lc = pd.read_csv(r"D:\Shraddha\Data\corr_sfc_esc_all_atlas_lc.csv", header = None).values[:, 1]
corr_efc_esc = pd.read_csv(r"D:\Shraddha\Empirical_correlations_all_atlas\Atlas_S200.csv", header = None).values[:, 1]

pheno_data = pd.read_csv(r"D:\Shraddha\Data\unrestricted_shraddhajain13_2_4_2021_6_34_39.csv").values
pheno_subject_num = pheno_data[:, 0]
brain_size_list_full = pheno_data[:, 192] #list of all subject's brain sizes in the phenotypical data
gender_list = pheno_data[:, 3]

def rearr(list_1): #function to rearrange the values in the order in which the subjects were simulated
    list_2 = []
    for i in range(272):
        ind = np.where(pheno_subject_num == sub_num_list[i])[0][0]
        list_2.append(list_1[ind])
    return(list_2)

brain_size_list_ordered = rearr(brain_size_list_full)
gender_list_filtered = rearr(gender_list)

phen_data_rest = pd.read_csv(r"D:\Shraddha\Data\RESTRICTED_shraddhajain13_3_3_2021_3_41_5.csv").values
age_phen = phen_data_rest[:, 1]

age_filtered = rearr(age_phen)

d = {'Subject ID': sub_num_list, 'Age': age_filtered, 'Sex': gender_list_filtered, 'Brain size - TIV': brain_size_list_ordered, 'corr(eFC, eSC)': corr_efc_esc, 'corr(sFC, eFC) - Phase Oscillator model': corr_sfc_efc_phase, 'corr(sFC, eFC) - LC model': corr_sfc_efc_lc, 'corr(sFC, eSC) - Phase Oscillator model:': corr_sfc_esc_phase, 'corr(sFC, eSC) - LC model': corr_sfc_esc_lc}
df = pd.DataFrame(data = d)
df.to_csv(r"D:\Shraddha\Data\file_for_S200.csv", index = False)