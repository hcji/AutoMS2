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
    files = [f for f in files if f.endswith('mzML')]
    data_columns = [f.split('.')[0] for f in files]
    
    msdial = raw.iloc[4:,:-2]
    msdial.columns = raw.iloc[3,:-2]
    keep_columns = ['Average Rt(min)', 'Average Mz', 'Adduct type', 'MS/MS spectrum', 
                    'Metabolite name', 'INCHIKEY', 'SMILES', 'Dot product'] + data_columns
    msdial = msdial.loc[:, keep_columns]
    msdial.columns = ['RT', 'MZ', 'Adduct', 'Tandem_MS', 
                      'Annotated Name', 'InChIKey', 'SMILES', 'Matching Score'] + files
    msdial['RT'] = np.round(60 * msdial['RT'].values.astype(float) ,3)
    msdial.loc[:,files] = msdial.loc[:,files].astype(float).replace(0, np.nan)
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
    msdial = msdial.reset_index(drop = True) 
    return msdial


def load_mzmine_result(data_path, mzmine_path):
    raw = pd.read_csv(mzmine_path, low_memory=False)
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith('mzML')]
    data_columns = ['datafile:{}:height'.format(f) for f in files]
    
    keep_columns = ['rt', 'mz'] + data_columns
    output = raw.loc[:, keep_columns]
    output.columns = ['RT', 'MZ'] + files
    output['Annotated Name'] = None
    output['InChIKey'] = None
    output['SMILES'] = None
    output['Matching Score'] = None
    return output


def load_xcms_result(data_path, xcms_path):
    raw = pd.read_csv(xcms_path, low_memory=False)
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith('mzML')]
    data_columns = [f.split('.')[0] for f in files]
    
    keep_columns = ['rtmed', 'mzmed'] + data_columns
    output = raw.loc[:, keep_columns]
    output.columns = ['RT', 'MZ'] + files
    output['Annotated Name'] = None
    output['InChIKey'] = None
    output['SMILES'] = None
    output['Matching Score'] = None
    return output

