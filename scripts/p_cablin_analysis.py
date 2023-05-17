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
automs_hpic_pos = automs.AutoMSData(ion_mode = 'positive')
automs_hpic_pos.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/POS")
automs_hpic_pos.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_pos.match_features()
automs_hpic_pos.match_features_with_ms2()
automs_hpic_pos.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")

# negative
automs_hpic_neg = automs.AutoMSData(ion_mode = 'negative')
automs_hpic_neg.load_files("E:/Data/Guanghuoxiang/Convert_files_mzML/NEG")
automs_hpic_neg.find_features(min_intensity = 20000, max_items = 100000)
automs_hpic_neg.match_features()
automs_hpic_neg.match_features_with_ms2()
automs_hpic_neg.save_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_negative.project")




# analyze positive mode
group_info = {'PX_L': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                       'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
              'ZX_L': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                       'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
              'NX_L': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                       'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML'],
              }

automs_hpic_pos = automs.AutoMSData(ion_mode = 'positive')
automs_hpic_pos.load_project("E:/Data/Guanghuoxiang/AutoMS_processing/guanghuoxiang_hpic_positive.project")
automs_hpic_pos = automs_hpic_pos.export_features()



