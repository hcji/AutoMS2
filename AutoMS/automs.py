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
from AutoMS import imputer
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
    
    
    def impute_missing_features(self, method = 'KNN', **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        files = list(self.peaks.keys())
        x = self.feature_table.loc[:,files]
        imp = imputer.Imputer(x, None)
        if method == 'Low value':
            x_imp = imp.fill_with_low_value()
        elif method == 'Mean':
            x_imp = imp.fill_with_mean_value()
        elif method == 'Median':
            x_imp = imp.fill_with_median_value()
        elif method == 'KNN':
            x_imp = imp.fill_with_knn_imputer(**args)
        elif method == 'Iterative RF':
            x_imp = imp.fill_with_iterative_RF(**args)
        elif method == 'Iterative BR':
            x_imp = imp.fill_with_iterative_BR(**args)
        elif method == 'Iterative SVR':
            x_imp = imp.fill_with_iterative_SVR(**args)
        else:
            raise ValueError(f"Invalid imputation method: {method}")
        self.feature_table.loc[:,files] = x_imp[0]
    
    
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
        
    
    def perform_PLSDA(self, group_info = None, n_components=2, **args):
        if self.feature_table is None:
            raise ValueError('Please match peak first')
        if group_info is not None:
            files_keep = [value for key in group_info for value in group_info[key]]
            x = self.feature_table.loc[:,files_keep].T
            y = [key for key in group_info.keys() for f in files_keep if f in group_info[key]]
        else:
            raise ValueError('Please input group information')
        
        plsda = analysis.PLSDA(x, y, n_components=n_components)
        plsda.scale_data(**args)
        plsda.perform_PLSDA(**args)
        plsda.plot_2D(**args)
        plsda.leave_one_out_test(**args)
        plsda.perform_permutation_test(**args)
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
    
    
    def perform_T_Test(self):
        pass
    
    
    def select_biomarker(self, criterion = {'T_Test': 0.05, 'PLS_VIP': 1.0}):
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
    automs.impute_missing_features()
    automs.match_feature_with_ms2()
    automs.export_ms2_mgf('guanghuoxiang_tandem_ms.mgf')
    automs.save_project('guanghuoxiang.project')
    
    data_path = "E:/Data/Chuanxiong"
    automs = AutoMS(data_path)
    automs.load_project('chuanxiong.project')
    group_info = {'G': ['G1.mzML', 'G2.mzML', 'G3.mzML', 'G4.mzML', 'G5.mzML', 'G6.mzML', 'G7.mzML', 'G8.mzML'],
                  'X': ['X1.mzML', 'X2.mzML', 'X3.mzML', 'X4.mzML', 'X5.mzML', 'X6.mzML', 'X7.mzML', 'X8.mzML'],
                  'Y': ['Y1.mzML', 'Y2.mzML', 'Y3.mzML', 'Y4.mzML', 'Y5.mzML', 'Y6.mzML', 'Y7.mzML']}
    automs.perform_dimensional_reduction(group_info = group_info, method = 'PCA')
    
    