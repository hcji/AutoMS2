# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:04:55 2022

@author: DELL
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

from AutoMS import hpic
from AutoMS import peakeval
from AutoMS import matching
from AutoMS import tandem
from AutoMS import deepmass
from AutoMS import analysis


class AutoMS:
    def __init__(self, data_path, ion_mode = 'positive'):
        """
        Arguments:
            data_path: string
                path to the dataset locally
        """
        self.data_path = data_path
        self.ion_mode = ion_mode
        self.peaks = None
        self.feature_table = None
        self.biomarker_list = []
    
    
    def find_features(self, min_intensity, mass_inv = 1, rt_inv = 30, min_snr = 3, max_items = 50000):
        """
        Arguments:
            min_snr: float
                minimum signal noise ratio
            mass_inv: float
                minimum interval of the m/z values
            rt_inv: float
                minimum interval of the retention time
            min_intensity: string
                minimum intensity of a peak.
        """
        output = {}
        files = os.listdir(self.data_path)
        for i, f in enumerate(files):
            print('processing {}, {}/{} files, set maximum {} ion traces'.format(f, 1+i, len(files), max_items))
            peaks, pics = hpic.hpic(os.path.join(self.data_path, f), 
                                    min_intensity = min_intensity, 
                                    min_snr = min_snr, 
                                    mass_inv = mass_inv, 
                                    rt_inv = rt_inv,
                                    max_items = max_items)
            output[f] = {'peaks': peaks, 'pics': pics}
        self.peaks = output
        
    
    def evaluate_features(self):
        if self.peaks is None:
            raise ValueError('Please find peak first')
        for f, vals in self.peaks.items():
            peak = vals['peaks']
            pic = vals['pics']
            score = peakeval.evaluate_peaks(peak, pic)
            self.peaks[f]['peaks']['score'] = score 
    
    
    def match_features(self, method = 'simple', mz_tol = 0.01, rt_tol = 15, min_frac = 0.5):
        if self.peaks is None:
            raise ValueError('Please find peak first')
        linker = matching.FeatureMatching(self.peaks)
        if method == 'simple':
            linker.simple_matching(mz_tol = mz_tol, rt_tol = rt_tol)
        else:
            raise IOError('Invalid Method')
        self.feature_table = linker.feature_filter(min_frac = min_frac)
        self.feature_table['Ionmode'] = self.ion_mode
    
        
    def preprocessing(self, impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, qc_samples = None, group_info = None, **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        files = list(self.peaks.keys())
        x = self.feature_table.loc[:,files]
        preprocessor = analysis.Preprocessing(x)
        x_prep = preprocessor.one_step(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, qc_samples = qc_samples, group_info = group_info, **args)
        self.feature_table.loc[:,files] = x_prep
    
    
    def match_feature_with_ms2(self, mz_tol = 0.01, rt_tol = 15):
        files = [os.path.join(self.data_path, f) for f in list(self.peaks.keys())]
        spectrums = tandem.load_tandem_ms(files)
        spectrums = tandem.cluster_tandem_ms(spectrums, mz_tol = mz_tol, rt_tol = rt_tol)
        self.feature_table = tandem.feature_spectrum_matching(self.feature_table, spectrums, mz_tol = mz_tol, rt_tol = rt_tol)

    
    def export_ms2_mgf(self, save_path):
        deepmass.export_to_mgf(self.feature_table, save_path)
    
    
    def load_deepmass(self):
        pass
    
    
    def perform_ms2_network(self):
        pass
    
    
    def perform_dimensional_reduction(self, group_info = None, method = 'PCA', **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        files = list(self.peaks.keys())

        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = self.feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            x = self.feature_table.loc[:,files].T
            y = np.repeat('Samples', len(files))
            
        DRAnal = analysis.Dimensional_Reduction(x, y)
        DRAnal.scale_data(**args)
        if method == 'PCA':
            DRAnal.perform_PCA(**args)
        elif method == 'tSNE':
            DRAnal.perform_tSNE(**args)
        elif method == 'uMAP':
            DRAnal.perform_uMAP(**args)
        else:
            raise IOError('Invalid Method')
        DRAnal.plot_2D(**args)
        
    
    def perform_PLSDA(self, group_info = None, n_components=2, n_permutations = 1000):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = self.feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        plsda = analysis.PLSDA(x, y, n_components = n_components)
        plsda.scale_data()
        plsda.perform_PLSDA()
        plsda.plot_2D()
        plsda.leave_one_out_test()
        plsda.perform_permutation_test(n_permutations = n_permutations)
        self.feature_table['PLS_VIP'] = plsda.get_VIP()        
    
    
    def perform_RandomForest(self, group_info = None, **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = self.feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        rf = analysis.RandomForest(x, y)
        rf.scale_data(**args)
        rf.perform_RF(**args)
        rf.out_of_bag_score()
        self.feature_table['RF_VIP'] = rf.get_VIP() 
    
    
    def perform_T_Test(self, group_info = None, **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = self.feature_table.loc[:,files_keep]
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        t_test = analysis.T_Test(x, y)
        t_test.perform_t_test()
        t_test.calc_fold_change()
        t_test.perform_multi_test_correlation(**args)
        t_test.plot_volcano()
        self.feature_table['T_Test_P_{}'.format('_'.join(group_info.keys()))] = t_test.p_values
    
    
    def select_biomarker(self, criterion = {'PLS_VIP': 1.0}):
        pass
    
    
    def perform_heatmap(self):
        pass
    
    
    def perform_enrichment_analysis(self):
        pass
    
    
    def perform_biomarker_analysis(self):
        pass
    
    
    def save_project(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    
    def load_project(self, save_path):
        with open(save_path, 'rb') as f:
            obj_dict = pickle.load(f)
        self.__dict__.update(obj_dict)





if __name__ == '__main__':
    
    data_path = "E:/Data/Guanghuoxiang/Convert_files_mzML/POS"
    automs = AutoMS(data_path)
    automs.find_features(min_intensity = 20000, max_items = 100000)
    automs.match_features()
    
    data_path = "E:/Data/Guanghuoxiang/Convert_files_mzML/POS"
    automs = AutoMS(data_path)
    automs.load_project('guanghuoxiang.project')
    qc_samples = ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML']
    group_info = {'QC': ['HF1_1578259_CP_QC1.mzML', 'HF1_1578259_CP_QC2.mzML', 'HF1_1578259_CP_QC3.mzML', 'HF1_1578259_CP_QC4.mzML', 'HF1_1578259_CP_QC5.mzML'],
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
    
    automs.preprocessing(impute_method = 'KNN', outlier_threshold = 3, rsd_threshold = 0.3, 
                         qc_samples = qc_samples, group_info = group_info)
    automs.match_feature_with_ms2()
    automs.export_ms2_mgf('guanghuoxiang_tandem_ms.mgf')
    automs.save_project('guanghuoxiang.project')
    
    automs.perform_dimensional_reduction(group_info = group_info, method = 'tSNE')
    automs.perform_PLSDA(group_info = {'PX_L': ['HF1_CP1_FZTM230002472-1A.mzML','HF1_CP1_FZTM230002473-1A.mzML','HF1_CP1_FZTM230002474-1A.mzML',
                                                'HF1_CP1_FZTM230002475-1A.mzML', 'HF1_CP1_FZTM230002476-1A.mzML', 'HF1_CP1_FZTM230002477-1A.mzML'],
                                       'PX_S': ['HF1_CP2_FZTM230002478-1A.mzML', 'HF1_CP2_FZTM230002479-1A.mzML', 'HF1_CP2_FZTM230002480-1A.mzML',
                                                'HF1_CP2_FZTM230002481-1A.mzML','HF1_CP2_FZTM230002482-1A.mzML','HF1_CP2_FZTM230002483-1A.mzML']})
    automs.perform_RandomForest(group_info = group_info)