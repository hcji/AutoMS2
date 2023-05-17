# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:46:53 2023

@author: DELL
"""

import numpy as np
from AutoMS import automs

# group information
qc_samples_pos = ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML']
group_info_pos = {'QC': ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML'],
                  'PX_L': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                           'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
                  'PX_S': ['HF1_CP2_FZTM230002478-1A.mzML', 'HF1_CP2_FZTM230002479-1A.mzML', 'HF1_CP2_FZTM230002480-1A.mzML',
                           'HF1_CP2_FZTM230002481-1A.mzML','HF1_CP2_FZTM230002482-1A.mzML','HF1_CP2_FZTM230002483-1A.mzML'],
                  'ZX_L': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                           'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
                  'ZX_S': ['HF1_CP4_FZTM230002490-1A.mzML', 'HF1_CP4_FZTM230002491-1A.mzML', 'HF1_CP4_FZTM230002492-1A.mzML',
                           'HF1_CP4_FZTM230002493-1A.mzML', 'HF1_CP4_FZTM230002494-1A.mzML', 'HF1_CP4_FZTM230002495-1A.mzML'],
                  'NX_L': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                           'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML'],
                  'NX_S': ['HF1_CP6_FZTM230002502-1A.mzML', 'HF1_CP6_FZTM230002503-1A.mzML', 'HF1_CP6_FZTM230002504-1A.mzML',
                           'HF1_CP6_FZTM230002505-1A.mzML', 'HF1_CP6_FZTM230002506-1A.mzML', 'HF1_CP6_FZTM230002507-1A.mzML']
                  }

qc_samples_neg = ['HF1_1578259_CN_QC1.mzML', 'HF1_1578259_CN_QC2.mzML', 'HF1_1578259_CN_QC3.mzML', 'HF1_1578259_CN_QC4.mzML', 'HF1_1578259_CN_QC5.mzML']
group_info_neg = {'QC': ['HF1_1578259_CN_QC1.mzML', 'HF1_1578259_CN_QC2.mzML', 'HF1_1578259_CN_QC3.mzML', 'HF1_1578259_CN_QC4.mzML', 'HF1_1578259_CN_QC5.mzML'],
                  'PX_L': ['HF1_CN1_FZTM230002472-1A.mzML','HF1_CN1_FZTM230002473-1A.mzML','HF1_CN1_FZTM230002474-1A.mzML',
                           'HF1_CN1_FZTM230002475-1A.mzML', 'HF1_CN1_FZTM230002476-1A.mzML', 'HF1_CN1_FZTM230002477-1A.mzML'],
                  'PX_S': ['HF1_CN2_FZTM230002478-1A.mzML', 'HF1_CN2_FZTM230002479-1A.mzML', 'HF1_CN2_FZTM230002480-1A.mzML',
                           'HF1_CN2_FZTM230002481-1A.mzML','HF1_CN2_FZTM230002482-1A.mzML','HF1_CN2_FZTM230002483-1A.mzML'],
                  'ZX_L': ['HF1_CN3_FZTM230002484-1A.mzML', 'HF1_CN3_FZTM230002485-1A.mzML', 'HF1_CN3_FZTM230002486-1A.mzML',
                           'HF1_CN3_FZTM230002487-1A.mzML', 'HF1_CN3_FZTM230002488-1A.mzML', 'HF1_CN3_FZTM230002489-1A.mzML'],
                  'ZX_S': ['HF1_CN4_FZTM230002490-1A.mzML', 'HF1_CN4_FZTM230002491-1A.mzML', 'HF1_CN4_FZTM230002492-1A.mzML',
                           'HF1_CN4_FZTM230002493-1A.mzML', 'HF1_CN4_FZTM230002494-1A.mzML', 'HF1_CN4_FZTM230002495-1A.mzML'],
                  'NX_L': ['HF1_CN5_FZTM230002496-1A.mzML', 'HF1_CN5_FZTM230002497-1A.mzML', 'HF1_CN5_FZTM230002498-1A.mzML',
                           'HF1_CN5_FZTM230002499-1A.mzML', 'HF1_CN5_FZTM230002500-1A.mzML', 'HF1_CN5_FZTM230002501-1A.mzML'],
                  'NX_S': ['HF1_CN6_FZTM230002502-1A.mzML', 'HF1_CN6_FZTM230002503-1A.mzML', 'HF1_CN6_FZTM230002504-1A.mzML',
                           'HF1_CN6_FZTM230002505-1A.mzML', 'HF1_CN6_FZTM230002506-1A.mzML', 'HF1_CN6_FZTM230002507-1A.mzML']
                  }

# data preprocessing
# positive
automs_hpic_pos = automs.AutoMS(ion_mode = 'positive')
automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
automs_hpic_pos.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_pos.match_features()
# automs_hpic_pos.match_features_with_external_annotation("E:/Data/Guanghuoxiang/meta_intensity_pos_classfire.csv")
automs_hpic_pos.match_features_with_ms2()
automs_hpic_pos.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                              qc_samples = qc_samples_pos, group_info = group_info_pos)
automs_hpic_pos.search_library("Library/references_spectrums_positive.pickle")
automs_hpic_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")

# negative
automs_hpic_neg = automs.AutoMS(ion_mode = 'negative')
automs_hpic_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
automs_hpic_neg.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_neg.match_features()
# automs_hpic_neg.match_features_with_external_annotation("E:/Data/Guanghuoxiang/meta_intensity_neg_classfire.csv")
automs_hpic_neg.match_features_with_ms2()
automs_hpic_neg.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                              qc_samples = qc_samples_neg, group_info = group_info_neg)
automs_hpic_neg.search_library("Library/references_spectrums_negative.pickle")
automs_hpic_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")


# ms-dial
# positive
automs_msdial_pos = automs.AutoMS(ion_mode = 'positive')
automs_msdial_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
automs_msdial_pos.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Positive_Height_0_2023423819.txt")
automs_msdial_pos.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                            qc_samples = qc_samples_pos, group_info = group_info_pos)
automs_msdial_pos.match_features_with_ms2()
automs_msdial_pos.search_library("Library/references_spectrums_positive.pickle")
automs_msdial_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_positive.project")


# negative
automs_msdial_neg = automs.AutoMS(ion_mode = 'negative')
automs_msdial_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
automs_msdial_neg.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Negative_Height_0_202351689.txt")
automs_msdial_neg.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                                qc_samples = qc_samples_neg, group_info = group_info_neg)
automs_msdial_neg.match_features_with_ms2()
automs_msdial_neg.search_library("Library/references_spectrums_negative.pickle")
automs_msdial_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_negative.project")



# analyze positive mode
group_info = {'PX_L': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                       'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
              'ZX_L': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                       'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
              'NX_L': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                       'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML'],
              }

automs_hpic_pos = automs.AutoMS(ion_mode = 'positive')
automs_hpic_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")
automs_msdial_pos = automs.AutoMS(ion_mode = 'positive')
automs_msdial_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_positive.project")

automs_hpic_pos.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial_pos.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic_pos.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)
automs_msdial_pos.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)

automs_hpic_pos.perform_RandomForest(group_info = group_info, n_estimators = 500)
automs_msdial_pos.perform_RandomForest(group_info = group_info, n_estimators = 500)

pls_vips = automs_msdial_pos.feature_table_annotated['PLS_VIP'].values
pls_vip_thres = -np.sort(-pls_vips)[50]
rf_vips = automs_msdial_pos.feature_table_annotated['RF_VIP'].values
rf_vip_thres = -np.sort(-rf_vips)[50]

automs_msdial_pos.select_biomarker(criterion = {'PLS_VIP': ['>', pls_vip_thres], 'RF_VIP': ['>', rf_vip_thres]}, combination = 'intersection')
automs_msdial_pos.perform_heatmap(group_info = group_info, hide_xticks = True, hide_ytick = True)


# analyze negative mode 
group_info = {'PX_L': ['HF1_CN1_FZTM230002472-1A.mzML','HF1_CN1_FZTM230002473-1A.mzML','HF1_CN1_FZTM230002474-1A.mzML',
                       'HF1_CN1_FZTM230002475-1A.mzML', 'HF1_CN1_FZTM230002476-1A.mzML', 'HF1_CN1_FZTM230002477-1A.mzML'],
              'ZX_L': ['HF1_CN3_FZTM230002484-1A.mzML', 'HF1_CN3_FZTM230002485-1A.mzML', 'HF1_CN3_FZTM230002486-1A.mzML',
                       'HF1_CN3_FZTM230002487-1A.mzML', 'HF1_CN3_FZTM230002488-1A.mzML', 'HF1_CN3_FZTM230002489-1A.mzML'],
              'NX_L': ['HF1_CN5_FZTM230002496-1A.mzML', 'HF1_CN5_FZTM230002497-1A.mzML', 'HF1_CN5_FZTM230002498-1A.mzML',
                       'HF1_CN5_FZTM230002499-1A.mzML', 'HF1_CN5_FZTM230002500-1A.mzML', 'HF1_CN5_FZTM230002501-1A.mzML'],
              }

automs_hpic_neg = automs.AutoMS(ion_mode = 'negative')
automs_hpic_neg.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")
automs_msdial_neg = automs.AutoMS(ion_mode = 'positive')
automs_msdial_neg.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_negative.project")

automs_hpic_neg.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial_neg.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic_neg.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)
automs_msdial_neg.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)

automs_hpic_neg.perform_RandomForest(group_info = group_info, n_estimators = 500)
automs_msdial_neg.perform_RandomForest(group_info = group_info, n_estimators = 500)

pls_vips = automs_msdial_neg.feature_table_annotated['PLS_VIP'].values
pls_vip_thres = -np.sort(-pls_vips)[50]
rf_vips = automs_msdial_neg.feature_table_annotated['RF_VIP'].values
rf_vip_thres = -np.sort(-rf_vips)[50]

automs_msdial_neg.select_biomarker(criterion = {'PLS_VIP': ['>', pls_vip_thres], 'RF_VIP': ['>', rf_vip_thres]}, combination = 'union')
automs_msdial_neg.perform_heatmap(group_info = group_info, hide_xticks = True, hide_ytick = True)