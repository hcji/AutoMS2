# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:42:40 2023

@author: DELL
"""


import pickle
import numpy as np
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

from AutoMS.SpectralEntropy import similarity as calc_similarity


class SpecLib:
    def __init__(self, library_path):
        print('load database...')
        with open(library_path, 'rb') as file:
            self.lib = pickle.load(file)
        self.precursor_mzs = np.array([s.get('precursor_mz') for s in self.lib]).astype(float)
        self.adducts = np.array([s.get('adduct') for s in self.lib]).astype(str)
        self.feature_table = None
    
    
    def search(self, feature_table, method = 'entropy', ms1_da=0.01, ms2_da=0.05, threshold = 0.5, synonyms = True):
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
                if synonyms:
                    feature_table.loc[i, 'Annotated Name'] = self.get_synonyms(lib[k].get('smiles'))
                else:
                    feature_table.loc[i, 'Annotated Name'] = lib[k].get('compound_name')
                feature_table.loc[i, 'InChIKey'] = lib[k].get('inchikey')
                feature_table.loc[i, 'SMILES'] = lib[k].get('smiles')
                feature_table.loc[i, 'Matching Score'] = np.max(scores)
                if lib[k].get('class') is None:
                    feature_table.loc[i, 'Class'] = self.predict_class(lib[k].get('smiles'))
                else:
                    feature_table.loc[i, 'Class'] = lib[k].get('class')
        self.feature_table = feature_table
        return self.feature_table
    
    
    def predict_class(self, smi, timeout = 60):
        url = 'https://npclassifier.ucsd.edu/classify?smiles={}'.format(smi)
        try:
            response = requests.get(url, timeout=timeout)
            soup = BeautifulSoup(response.content, "html.parser") 
            output = json.loads(str(soup))['class_results']
        except:
            return None
        if len(output) >= 1:
            return output[0]
        else:
            return None
    
    def get_synonyms(self, smi, timeout = 60):
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/Synonyms/json'.format(smi)
        try:
            response = requests.get(url, timeout=timeout)
            soup = BeautifulSoup(response.content, "html.parser") 
            output = json.loads(str(soup))['InformationList']['Information'][0]['Synonym']
        except:
            return None
        if len(output) >= 1:
            return output[0]
        else:
            return None
    
    