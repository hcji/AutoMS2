# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:32:52 2023

@author: DELL
"""

import warnings
warnings.filterwarnings("ignore")
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
        spectrum.set('compound_name', 'coumpound_{}'.format(i))
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
    save_as_mgf(spectrums, save_path)
    print('Finished')
    