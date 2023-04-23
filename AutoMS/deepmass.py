# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:32:52 2023

@author: DELL
"""

import os
import numpy as np
import pandas as pd
import matchms.filtering as msfilters
from matchms.exporting import save_as_mgf, save_as_msp
from tqdm import tqdm

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
    return s


def export_to_mgf(feature_table, save_path):
    spectrums = []
    for i in tqdm(feature_table.index):
        spectrum = feature_table.loc[i, 'Tandem_MS']
        if spectrum is None:
            continue
        spectrum.set('compound_name', 'compound_{}'.format(i))
        spectrum.metadata['title'] = spectrum.metadata['compound_name']
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
        if 'Adduct' not in feature_table.columns:
            spectrum.set('PRECURSORTYPE', '[M+H]+')
        else:
            spectrum.set('PRECURSORTYPE', feature_table.loc[i, 'Adduct'])
        if 'Ionmode' not in feature_table.columns:
            spectrum.set('ionmode', 'Positive')
        else:
            spectrum.set('ionmode', feature_table.loc[i, 'Ionmode'])
        spectrum = spectrum_processing(spectrum)
        save_filename = os.path.join(save_path, '{}.msp'.format(i))
        if os.path.exists(save_filename):
            os.remove(save_filename)
        save_as_msp([spectrum], save_filename)
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


def refine_annotated_table(feature_table, value_columns):
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

