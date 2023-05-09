# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 07:37:46 2023

@author: hcji
"""

from AutoMS import automs

automs_hpic = automs.AutoMS()
automs_hpic.load_project("E:/guanghuoxiang.project")

automs_msdial = automs.AutoMS()
automs_msdial.load_project("E:/guanghuoxiang_msdial.project")

group_info = {'PX_L': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                       'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
              'ZX_L': ['HF1_CP3_FZTM230002484-1A.mzML', 'HF1_CP3_FZTM230002485-1A.mzML', 'HF1_CP3_FZTM230002486-1A.mzML',
                       'HF1_CP3_FZTM230002487-1A.mzML', 'HF1_CP3_FZTM230002488-1A.mzML', 'HF1_CP3_FZTM230002489-1A.mzML'],
              'NX_L': ['HF1_CP5_FZTM230002496-1A.mzML', 'HF1_CP5_FZTM230002497-1A.mzML', 'HF1_CP5_FZTM230002498-1A.mzML',
                       'HF1_CP5_FZTM230002499-1A.mzML', 'HF1_CP5_FZTM230002500-1A.mzML', 'HF1_CP5_FZTM230002501-1A.mzML'],
              }

automs_hpic.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)
automs_msdial.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)

automs_hpic.perform_RandomForest(group_info = group_info, n_estimators = 500)
automs_msdial.perform_RandomForest(group_info = group_info, n_estimators = 500)

automs_hpic.select_biomarker(criterion = {'PLS_VIP': ['>', 1.66478]})
automs_hpic.perform_heatmap(group_info = group_info, hide_xticks = True, hide_ytick = True)





group_info = {'PX_S': ['HF1_CP2_FZTM230002478-1A.mzML', 'HF1_CP2_FZTM230002479-1A.mzML', 'HF1_CP2_FZTM230002480-1A.mzML',
                       'HF1_CP2_FZTM230002481-1A.mzML','HF1_CP2_FZTM230002482-1A.mzML','HF1_CP2_FZTM230002483-1A.mzML'],
              'ZX_S': ['HF1_CP4_FZTM230002490-1A.mzML', 'HF1_CP4_FZTM230002491-1A.mzML', 'HF1_CP4_FZTM230002492-1A.mzML',
                       'HF1_CP4_FZTM230002493-1A.mzML', 'HF1_CP4_FZTM230002494-1A.mzML', 'HF1_CP4_FZTM230002495-1A.mzML'],
              'NX_S': ['HF1_CP6_FZTM230002502-1A.mzML', 'HF1_CP6_FZTM230002503-1A.mzML', 'HF1_CP6_FZTM230002504-1A.mzML',
                       'HF1_CP6_FZTM230002505-1A.mzML', 'HF1_CP6_FZTM230002506-1A.mzML', 'HF1_CP6_FZTM230002507-1A.mzML']
              }

automs_hpic.perform_PLSDA(group_info = group_info, n_components = 3)
automs_msdial.perform_PLSDA(group_info = group_info, n_components = 3)

automs_hpic.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)
automs_msdial.perform_GradientBoost(group_info = group_info, model = 'XGBoost', n_estimators = 1000, max_depth = 10, max_leaves = 0)

automs_hpic.perform_RandomForest(group_info = group_info, n_estimators = 500)
automs_msdial.perform_RandomForest(group_info = group_info, n_estimators = 500)

automs_hpic.select_biomarker(criterion = {'PLS_VIP': ['>', 1.78792], 'RF_VIP': ['>', 0.00396217]})
automs_hpic.perform_heatmap(group_info = group_info, hide_ytick = False)


