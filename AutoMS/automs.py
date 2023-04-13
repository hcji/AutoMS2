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


class AutoMS:
    def __init__(self, data_path):
        """
        Arguments:
            data_path: string
                path to the dataset locally
        """
        self.data_path = data_path
        self.peaks = None
        self.feature_table = None
    
    
    def find_peaks(self, min_intensity, mass_inv = 1, rt_inv = 30, min_snr = 3, max_items = 50000):
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
        
    
    def evaluate_peaks(self):
        if self.peaks is None:
            raise ValueError('Please find peak first')
        for f, vals in self.peaks.items():
            peak = vals['peaks']
            pic = vals['pics']
            score = peakeval.evaluate_peaks(peak, pic)
            self.peaks[f]['peaks']['score'] = score 
    
    
    def match_peaks(self, method = 'simple', mz_tol = 0.01, rt_tol = 15, min_frac = 0.5):
        if self.peaks is None:
            raise ValueError('Please find peak first')
        linker = matching.FeatureMatching(self.peaks)
        if method == 'simple':
            linker.simple_matching(mz_tol = mz_tol, rt_tol = rt_tol)
        else:
            raise IOError('Invalid Method')
        self.feature_table = linker.feature_filter(min_frac = min_frac)
    
    
    def impute_missing_value(self, method = 'KNN', **args):
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
    
    
    def match_with_ms2(self, mz_tol = 0.01, rt_tol = 15):
        files = [os.path.join(self.data_path, f) for f in list(self.peaks.keys())]
        spectrums = tandem.load_tandem_ms(files)
        spectrums = tandem.cluster_tandem_ms(spectrums, mz_tol = mz_tol, rt_tol = rt_tol)
        self.feature_table = tandem.feature_spectrum_matching(self.feature_table, spectrums, mz_tol = mz_tol, rt_tol = rt_tol)

        
    def perform_deisotope(self):
        pass
    
    
    def export_ms2_mgf(self):
        pass
    
    
    def load_deepmass(self):
        pass
    
    
    def perform_ms2_network(self):
        pass
    
    
    def perform_PCA(self):
        pass
    
    
    def perform_PLSDA(self):
        pass
    
    
    def perform_T_Test(self):
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
    
    data_path = "E:/Data/Chuanxiong"
    automs = AutoMS(data_path)
    automs.find_peaks(min_intensity = 20000, max_items = 100000)
    automs.match_peaks()
    automs.impute_missing_value()
    automs.save_project('chuanxiong.project')
    
    
    automs = AutoMS(data_path)
    automs.load_project('chuanxiong.project')
    
    