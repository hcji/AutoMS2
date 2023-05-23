# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:46:53 2023

@author: DELL
"""

import numpy as np
from AutoMS import automs

# prepare data
# positive
automs_hpic_pos = automs.AutoMSData(ion_mode = 'positive')
automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
automs_hpic_pos.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_pos.match_features()
automs_hpic_pos.match_features_with_ms2()
automs_hpic_pos.search_library("Library/references_spectrums_positive.pickle")
automs_hpic_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")

# negative
automs_hpic_neg = automs.AutoMSData(ion_mode = 'negative')
automs_hpic_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
automs_hpic_neg.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_neg.match_features()
automs_hpic_neg.match_features_with_ms2()
automs_hpic_neg.search_library("Library/references_spectrums_negative.pickle")
automs_hpic_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")

# load msdial result
automs_msdial_pos = automs.AutoMS(ion_mode = 'positive')
automs_msdial_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
automs_msdial_pos.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Positive_Height_0_2023423819.txt")
automs_msdial_pos.match_features_with_ms2()
automs_msdial_pos.search_library("Library/references_spectrums_positive.pickle")
automs_msdial_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_positive.project")

automs_msdial_neg = automs.AutoMS(ion_mode = 'negative')
automs_msdial_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
automs_msdial_neg.load_msdial("E:/Data/Guanghuoxiang/MSDIAL_processing/Negative_Height_0_202351689.txt")
automs_msdial_neg.match_features_with_ms2()
automs_msdial_neg.search_library("Library/references_spectrums_negative.pickle")
automs_msdial_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_negative.project")


# preprocess feature table
# update name
namelist = ['QC-{}'.format(i) for i in range(1,6)]
namelist += ['PX-L-{}'.format(i) for i in range(1,7)]
namelist += ['PX-S-{}'.format(i) for i in range(1,7)]
namelist += ['ZX-L-{}'.format(i) for i in range(1,7)]
namelist += ['ZX-S-{}'.format(i) for i in range(1,7)]
namelist += ['NX-L-{}'.format(i) for i in range(1,7)]
namelist += ['NX-S-{}'.format(i) for i in range(1,7)]

# group information 
qc_samples = ['QC-{}'.format(i) for i in range(1,6)]
group_info = {'QC': ['QC-{}'.format(i) for i in range(1,6)],
              'PX_L': ['PX-L-{}'.format(i) for i in range(1,7)],
              'PX_S': ['PX-S-{}'.format(i) for i in range(1,7)],
              'ZX_L': ['ZX-L-{}'.format(i) for i in range(1,7)],
              'ZX_S': ['ZX-S-{}'.format(i) for i in range(1,7)],
              'NX_L': ['NX-L-{}'.format(i) for i in range(1,7)],
              'NX_S': ['NX-S-{}'.format(i) for i in range(1,7)]
              }

automs_hpic_pos = automs.AutoMSData(ion_mode = 'positive')
automs_hpic_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")
automs_hpic_pos = automs_hpic_pos.export_features()
automs_hpic_pos.update_sample_name(namelist)

automs_hpic_neg = automs.AutoMSData(ion_mode = 'negative')
automs_hpic_neg.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")
automs_hpic_neg = automs_hpic_neg.export_features()
automs_hpic_neg.update_sample_name(namelist)

automs_msdial_pos = automs.AutoMSData(ion_mode = 'positive')
automs_msdial_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_positive.project")
automs_msdial_pos = automs_msdial_pos.export_features()
automs_msdial_pos.update_sample_name(namelist)

automs_msdial_neg = automs.AutoMSData(ion_mode = 'negative')
automs_msdial_neg.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_msdial_negative.project")
automs_msdial_neg = automs_msdial_neg.export_features()
automs_msdial_neg.update_sample_name(namelist)


# merge positive and negative
automs_hpic_feat = automs.AutoMSFeature()
automs_hpic_feat.append_feature_table(automs_hpic_pos)
automs_hpic_feat.append_feature_table(automs_hpic_neg)
automs_hpic_feat.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                               qc_samples = qc_samples, group_info = group_info)
automs_hpic_feat.refine_annotated_table()

automs_msdial_feat = automs.AutoMSFeature()
automs_msdial_feat.append_feature_table(automs_msdial_pos)
automs_msdial_feat.append_feature_table(automs_msdial_neg)
automs_msdial_feat.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, min_frac = 0.5,
                                 qc_samples = qc_samples, group_info = group_info)
automs_msdial_feat.refine_annotated_table()


# dimensional reduction analysis
automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'PCA', annotated_only = False)
automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'tSNE', annotated_only = False)
automs_hpic_feat.perform_dimensional_reduction(group_info = group_info, method = 'uMAP', annotated_only = False)

automs_msdial_feat.perform_dimensional_reduction(group_info = group_info, method = 'PCA', annotated_only = False)
automs_msdial_feat.perform_dimensional_reduction(group_info = group_info, method = 'tSNE', annotated_only = False)
automs_msdial_feat.perform_dimensional_reduction(group_info = group_info, method = 'uMAP', annotated_only = False)


# analyze leaf samples
group_info = {'PX_L': ['PX-L-{}'.format(i) for i in range(1,7)],
              'ZX_L': ['ZX-L-{}'.format(i) for i in range(1,7)]+ ['NX-L-{}'.format(i) for i in range(1,7)]
              }
automs_hpic_feat.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial_feat.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic_feat.perform_GradientBoost(group_info = group_info, n_estimators = 500)
automs_msdial_feat.perform_GradientBoost(group_info = group_info, n_estimators = 500)

automs_hpic_feat.perform_RandomForest(group_info = group_info)
automs_msdial_feat.perform_RandomForest(group_info = group_info)

pls_vips = automs_hpic_feat.feature_table_annotated['PLS_VIP'].values
pls_vip_thres = -np.sort(-pls_vips)[30]
rf_vips = automs_hpic_feat.feature_table_annotated['RF_VIP'].values
rf_vip_thres = -np.sort(-rf_vips)[30]
automs_hpic_feat.select_biomarker(criterion = {'PLS_VIP': ['>', pls_vips], 'RF_VIP': ['>', rf_vip_thres]}, combination = 'union')
automs_hpic_feat.perform_heatmap(group_info = group_info, hide_xticks = False, hide_ytick = False)



group_info = {'PX_S': ['PX-S-{}'.format(i) for i in range(1,7)],
              'ZX_S': ['ZX-S-{}'.format(i) for i in range(1,7)],
              'NX_S': ['NX-S-{}'.format(i) for i in range(1,7)]
              }
automs_hpic_feat.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial_feat.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic_feat.perform_GradientBoost(group_info = group_info)
automs_msdial_feat.perform_GradientBoost(group_info = group_info)

automs_hpic_feat.perform_RandomForest(group_info = group_info, n_estimators = 500)
automs_msdial_feat.perform_RandomForest(group_info = group_info, n_estimators = 500)

pls_vips = automs_hpic_feat.feature_table_annotated['PLS_VIP'].values
pls_vip_thres = -np.sort(-pls_vips)[50]
rf_vips = automs_hpic_feat.feature_table_annotated['RF_VIP'].values
rf_vip_thres = -np.sort(-rf_vips)[50]
automs_hpic_feat.select_biomarker(criterion = {'PLS_VIP': ['>', pls_vip_thres], 'RF_VIP': ['>', rf_vip_thres]}, combination = 'intersection')
automs_hpic_feat.perform_heatmap(group_info = group_info, hide_xticks = False, hide_ytick = False)


