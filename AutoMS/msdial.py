# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:27:42 2023

@author: DELL
"""

import os
import numpy as np
import pandas as pd
from matchms import Spectrum

def load_msdial_result(data_path, msdial_path):
    raw = pd.read_csv(msdial_path, sep = '\t', low_memory=False)
    files = os.listdir(data_path)
    data_columns = [f.split('.')[0] for f in files]
    
    msdial = raw.iloc[4:,:-2]
    msdial.columns = raw.iloc[3,:-2]
    keep_columns = ['Average Rt(min)', 'Average Mz', 'Adduct type', 'MS/MS spectrum', 
                    'Metabolite name', 'INCHIKEY', 'SMILES', 'Dot product'] + data_columns
    msdial = msdial.loc[:, keep_columns]
    msdial.columns = ['RT', 'MZ', 'Adduct', 'Tandem_MS', 
                      'Annotated Name', 'InChIKey', 'SMILES', 'Matching Score'] + data_columns
    msdial['RT'] = np.round(60 * msdial['RT'].values.astype(float) ,3)
    msdial.loc[:,data_columns] = msdial.loc[:,data_columns].astype(float).replace(0, np.nan)
    msdial.loc[:,['Annotated Name']] = msdial.loc[:,['Annotated Name']].replace('Unknown', None)
    msdial.loc[:,['InChIKey', 'SMILES']] = msdial.loc[:,['InChIKey', 'SMILES']].astype(float).replace(np.nan, None)
    
    msmslist = []
    for i in msdial.index:
        s = str(msdial.loc[i, 'Tandem_MS'])
        precursor_mz = float(msdial.loc[i, 'MZ'])
        precursor_rt = float(msdial.loc[i, 'RT'])
        if s == 'nan':
            msmslist.append(None)
        else:
            s = s.split(' ')
            mz = np.array([float(ss.split(':')[0]) for ss in s if ':' in ss])
            intensity = np.array([float(ss.split(':')[1]) for ss in s if ':' in ss])      
            intensity /= (np.max(intensity) + 10 **-10)
            msmslist.append(Spectrum(mz=np.array(mz),
                                     intensities=np.array(intensity),
                                     metadata={'spectrum_id': 'spectrum_{}'.format(i),
                                               "precursor_mz": precursor_mz,
                                               "retention_time": precursor_rt}))
    msdial['Tandem_MS'] = msmslist
    return msdial
    