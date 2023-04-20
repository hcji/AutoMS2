# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:32:52 2023

@author: DELL
"""

import os
import numpy as np
import pandas as pd
import matchms.filtering as msfilters
from matchms.exporting import save_as_mgf
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


def link_to_deepmass(feature_table, deepmass_dir):
    print('load annotation results from deepmass dir')
    smiles, inchikeys, names = [], [], []
    for i in tqdm(feature_table.index):
        path = os.path.join(deepmass_dir, 'compound_{}.csv'.format(i))
        if os.path.exists(path):
            anno = pd.read_csv(path)
            if len(anno) > 1:
                [n, k, s] = anno.loc[0, ['Title', 'InChIKey', 'CanonicalSMILES']]
                smiles.append(s)
                inchikeys.append(k)
                names.append(n)
            else:
                smiles.append(None)
                inchikeys.append(None)
                names.append(None)
        else:
            smiles.append(None)
            inchikeys.append(None)
            names.append(None)
    feature_table['Annotated Name'] = names
    feature_table['InChIKey'] = inchikeys
    feature_table['CanonicalSMILES'] = smiles
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

