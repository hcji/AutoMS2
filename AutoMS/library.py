# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:42:40 2023

@author: DELL
"""


import pickle
import numpy as np
from tqdm import tqdm

from AutoMS.SpectralEntropy import similarity as calc_similarity


class SpecLib:
    def __init__(self, library_path):
        print('load database...')
        with open(library_path, 'rb') as file:
            self.lib = pickle.load(file)
        self.precursor_mzs = np.array([s.get('precursor_mz') for s in self.lib]).astype(float)
        self.adducts = np.array([s.get('adduct') for s in self.lib]).astype(str)
        self.feature_table = None
    
    
    def search(self, feature_table, method = 'entropy', ms1_da=0.01, ms2_da=0.05, threshold = 0.5):
        lib = self.lib
        precursor_mzs = self.precursor_mzs
        adducts = self.adducts
        print("search database...")
        for i in tqdm(feature_table.index):
            s = feature_table.loc[i, 'Tandem_MS']
            if s is None:
                continue
            
            mz = s.get('precursor_mz')
            if s.get('adduct') is None:
                k = np.abs(mz - precursor_mzs) < ms1_da
            else:
                k = np.logical_and(np.abs(mz - precursor_mzs) < ms1_da, s.get('adduct') == adducts)
            k = np.where(k)[0]
            if len(k) == 0:
                continue
            
            query = s.peaks.to_numpy.astype(np.float32)
            scores = []
            for j in k:
                reference = lib[j].peaks.to_numpy.astype(np.float32)
                scores.append(calc_similarity(query, reference, method=method, ms2_da=ms2_da))
            if np.max(scores) < threshold:
                continue
            else:
                k = k[np.argmax(scores)]                
                feature_table.loc[i, 'Annotated Name'] = lib[k].get('compound_name')
                feature_table.loc[i, 'InChIKey'] = lib[k].get('inchikey')
                feature_table.loc[i, 'SMILES'] = lib[k].get('smiles')
                feature_table.loc[i, 'Matching Score'] = np.max(scores)         
        self.feature_table = feature_table
    
    
    def refine_annotated_table(self, value_columns):
        feature_table = self.feature_table
        print('refine feature table with deepmass annotation')
        keep = []
        uni_comp = list(set(feature_table['InChIKey']))
        for key in tqdm(uni_comp):
            if key is None:
                continue
            wh = np.where(feature_table['InChIKey'] == key)[0]
            if len(wh) == 1:
                keep.append(wh[0])
            else:
                mean_vals = np.mean(feature_table.loc[wh, value_columns].values, axis = 1)
                keep.append(wh[np.argmax(mean_vals)])
        keep = np.sort(keep)
        feature_table_annotated = feature_table.loc[keep,:]
        return feature_table_annotated
    
    