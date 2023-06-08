# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:27:42 2023

@author: DELL
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matchms.filtering as msfilters
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp


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

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = msfilters.default_filters(s)
    if ('adduct_type' in s.metadata.keys()) and ('adduct' not in s.metadata.keys()):
        s.set('adduct', s.get('adduct_type'))
    s = msfilters.correct_charge(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.add_losses(s)
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    s = msfilters.reduce_to_number_of_peaks(s, n_max = 50, ratio_desired = 0.05)
    s = msfilters.remove_peaks_around_precursor_mz(s)
    return s


def export_to_mgf(feature_table, save_path):
    spectrums = []
    for i in tqdm(feature_table.index):
        spectrum = feature_table.loc[i, 'Tandem_MS']
        if spectrum is None:
            continue
        spectrum.set('compound_name', 'compound_{}'.format(i))
        spectrum.set('title', spectrum.metadata['compound_name'])
        if 'Adduct' not in feature_table.columns:
            spectrum.set('adduct', '[M+H]+')
        else:
            spectrum.set('adduct', feature_table.loc[i, 'Adduct'])
        if 'Ionmode' not in feature_table.columns:
            spectrum.set('ionmode', 'Positive')
        else:
            spectrum.set('ionmode', feature_table.loc[i, 'Ionmode'])
        spectrum = spectrum_processing(spectrum)
        spectrums.append(spectrum)
    if os.path.exists(save_path):
        os.remove(save_path)     
    save_as_mgf(spectrums, save_path)
    print('Finished')


def export_to_msp(feature_table, save_path):
    for i in tqdm(feature_table.index):
        spectrum = feature_table.loc[i, 'Tandem_MS']
        if spectrum is None:
            continue
        spectrum.set('compound_name', 'compound_{}'.format(i))
        spectrum.set('title', spectrum.metadata['compound_name'])
        if 'Adduct' not in feature_table.columns:
            spectrum.set('precursortype', '[M+H]+')
        else:
            spectrum.set('precursortype', feature_table.loc[i, 'Adduct'])
        if 'Ionmode' not in feature_table.columns:
            spectrum.set('ionmode', 'Positive')
        else:
            spectrum.set('ionmode', feature_table.loc[i, 'Ionmode'])
        spectrum = spectrum_processing(spectrum)
        save_filename = os.path.join(save_path, '{}.msp'.format(i))
        if os.path.exists(save_filename):
            os.remove(save_filename)
        save_as_msp([spectrum], save_filename)
        with open(save_filename, encoding = 'utf-8') as msp:
            lines = msp.readlines()
            lines = [l.replace('_', '') for l in lines]
            lines = [l.replace('ADDUCT', 'PRECURSORTYPE') for l in lines]
        with open(save_filename, 'w') as msp:
            msp.writelines(lines)
    print('Finished')


def link_to_deepmass(feature_table, deepmass_dir):
    print('load annotation results from deepmass dir')
    for i in tqdm(feature_table.index):
        path = os.path.join(deepmass_dir, 'compound_{}.csv'.format(i))
        if os.path.exists(path):
            anno = pd.read_csv(path)
            if len(anno) > 1:
                [n, k, s, c] = anno.loc[0, ['Title', 'InChIKey', 'CanonicalSMILES', 'Consensus Score']]      
                feature_table.loc[i, 'Annotated Name'] = n
                feature_table.loc[i, 'InChIKey'] = k
                feature_table.loc[i, 'SMILES'] = s
                feature_table.loc[i, 'Matching Score'] = c
            else:
                continue
        else:
            continue
    return feature_table