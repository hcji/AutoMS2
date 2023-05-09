# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:46:53 2023

@author: DELL
"""

from AutoMS import automs

# data information
data_path = "E:/Data/Guanghuoxiang/Convert_files_mzML/POS"
qc_samples = ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML']
group_info = {'QC': ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML'],
              'Leaf_PX': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                       'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
              'Stem_PX': ['HF1_CP2_FZTM230002478-1A.mzML', 'HF1_CP2_FZTM230002479-1A.mzML', 'HF1_CP2_FZTM230002480-1A.mzML',
                       'HF1_CP2_FZTM230002481-1A.mzML','HF1_CP2_FZTM230002482-1A.mzML','HF1_CP2_FZTM230002483-1A.mzML'],
              'Leaf_ZX': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                       'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
              'Stem_ZX': ['HF1_CP4_FZTM230002490-1A.mzML', 'HF1_CP4_FZTM230002491-1A.mzML', 'HF1_CP4_FZTM230002492-1A.mzML',
                       'HF1_CP4_FZTM230002493-1A.mzML', 'HF1_CP4_FZTM230002494-1A.mzML', 'HF1_CP4_FZTM230002495-1A.mzML'],
              'Leaf_NX': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                       'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML'],
              'Stem_NX': ['HF1_CP6_FZTM230002502-1A.mzML', 'HF1_CP6_FZTM230002503-1A.mzML', 'HF1_CP6_FZTM230002504-1A.mzML',
                       'HF1_CP6_FZTM230002505-1A.mzML', 'HF1_CP6_FZTM230002506-1A.mzML', 'HF1_CP6_FZTM230002507-1A.mzML']
              }

# Dimension Reduction Analysis
# HPIC
automs_hpic = automs.AutoMS(data_path)
automs_hpic.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang.project")
automs_hpic.perform_dimensional_reduction(group_info = group_info, method = 'PCA', annotated_only = False)
automs_hpic.perform_dimensional_reduction(group_info = group_info, method = 'tSNE', annotated_only = False)
automs_hpic.perform_dimensional_reduction(group_info = group_info, method = 'uMAP', annotated_only = False)


# MS-DIAL
'''
automs_msdial = automs.AutoMS(data_path)
automs_msdial.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Positive_Height_0_2023423819.txt")
automs_msdial.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                            qc_samples = qc_samples, group_info = group_info)
automs_msdial.match_features_with_ms2()
automs_msdial.search_library("Library/references_spectrums_positive.pickle")
automs_msdial.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial.project")
'''
automs_msdial = automs.AutoMS(data_path)
automs_msdial.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial.project")
automs_msdial.perform_dimensional_reduction(group_info = group_info, method = 'PCA', annotated_only = False)
automs_msdial.perform_dimensional_reduction(group_info = group_info, method = 'tSNE', annotated_only = False)
automs_msdial.perform_dimensional_reduction(group_info = group_info, method = 'uMAP', annotated_only = False)


# Multivariate Discriminant Analysis
group_info = {'Leaf_PX': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                       'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
              'Leaf_ZX': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                       'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
              'Leaf_NX': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                       'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML']
             }

automs_hpic.perform_PLSDA(group_info = group_info, n_components = 3, annotated_only = True)
automs_hpic.perform_RandomForest(group_info = group_info, annotated_only = True)
automs_hpic.perform_GradientBoost(model = 'LightGBM', group_info = group_info, annotated_only = True)

automs_msdial.perform_PLSDA(group_info = group_info, n_components = 3, annotated_only = True)
automs_msdial.perform_RandomForest(group_info = group_info, annotated_only = True)
automs_msdial.perform_GradientBoost(model = 'LightGBM', group_info = group_info, annotated_only = True)


group_info = {'Stem_PX': ['HF1_CP2_FZTM230002478-1A.mzML', 'HF1_CP2_FZTM230002479-1A.mzML', 'HF1_CP2_FZTM230002480-1A.mzML',
                       'HF1_CP2_FZTM230002481-1A.mzML','HF1_CP2_FZTM230002482-1A.mzML','HF1_CP2_FZTM230002483-1A.mzML'],
              'Stem_ZX': ['HF1_CP4_FZTM230002490-1A.mzML', 'HF1_CP4_FZTM230002491-1A.mzML', 'HF1_CP4_FZTM230002492-1A.mzML',
                       'HF1_CP4_FZTM230002493-1A.mzML', 'HF1_CP4_FZTM230002494-1A.mzML', 'HF1_CP4_FZTM230002495-1A.mzML'],
              'Stem_NX': ['HF1_CP6_FZTM230002502-1A.mzML', 'HF1_CP6_FZTM230002503-1A.mzML', 'HF1_CP6_FZTM230002504-1A.mzML',
                       'HF1_CP6_FZTM230002505-1A.mzML', 'HF1_CP6_FZTM230002506-1A.mzML', 'HF1_CP6_FZTM230002507-1A.mzML']
              }

automs_hpic.perform_PLSDA(group_info = group_info, n_components = 3, annotated_only = True)
automs_hpic.perform_RandomForest(group_info = group_info, annotated_only = True)
automs_hpic.perform_GradientBoost(group_info = group_info, annotated_only = True)

automs_msdial.perform_PLSDA(group_info = group_info, n_components = 3, annotated_only = True)
automs_msdial.perform_RandomForest(group_info = group_info, annotated_only = True)
automs_msdial.perform_GradientBoost(group_info = group_info, annotated_only = True)


