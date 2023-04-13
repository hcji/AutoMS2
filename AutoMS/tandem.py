# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:56:31 2023

@author: DELL
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from matchms import Spectrum
from matchms.importing import load_from_mzml


def load_tandem_ms(files):
    print('Loading tandem ms from files...\n')
    all_spectrums = []
    for f in tqdm(files):
        all_spectrums += [s for s in load_from_mzml(f)]
    all_spectrums = [s for s in all_spectrums if len(s.peaks.mz) > 5]
    return all_spectrums


def spectrum_to_vector(s, min_mz = 0, max_mz = 1000, scale = 0.1):    
    """
    Convert spectrum object to vector.
    Arguments:
        s: matchms::spectrum
        min_mz: float, start of mz value.
        max_mz: float, end of mz value.
        scale: float, scale of mz bin.
    Returns:
        Numpy array of spectrum.
    """
    bit = round((1 + max_mz - min_mz) / scale)
    vec = np.zeros(bit)
    if s is None:
        return vec
    else:
        k = np.logical_and(min_mz <= s.mz, s.mz <= max_mz)
        idx = np.round((s.mz[k] - min_mz) / scale).astype(int)
        val = s.intensities[k]
        vec[idx] = val
        vec = vec / (np.max(vec) + 10 ** -6)
        return vec


def cluster_tandem_ms(all_spectrums, mz_tol = 0.01, rt_tol = 15):
    mzs, rts, spectrums = [], [], []
    print('split spectrums based on rt and precursor mz...\n')
    for s in tqdm(all_spectrums):
        mz = s.get('precursor_mz')
        rt = s.get('scan_start_time')[0]
        if rt.unit_info == 'minute':
            rt = float(rt) * 60
        else:
            rt = float(rt)
        k1 = np.abs(mz - np.array(mzs)) < mz_tol
        k2 = np.abs(rt - np.array(rts)) < rt_tol
        kk = np.where(np.logical_and(k1, k2))[0]
        if len(kk) == 0:
            mzs.append(mz)
            rts.append(rt)
            spectrums.append([s])
            continue
        elif len(kk) > 1:
            k = kk[np.argmin(np.abs(mz - np.array(mzs)[kk]))]
        else:
            k = kk[0]
        
        n = len(spectrums[k])
        mzs[k] = (mzs[k] * n + mz) / (n+1)
        rts[k] = (rts[k] * n + rt) / (n+1)
        spectrums[k] = spectrums[k] + [s]
    
    print('cluster spectrums and generate consesus spectrum...\n')
    for i, spectrums_i in enumerate(tqdm(spectrums)):
        if len(spectrums_i) == 1:
            spectrums[i] = spectrums_i
            continue
        spectrums_vectors = [spectrum_to_vector(s) for s in spectrums_i]
        cos_distance = 1 - cosine_similarity(spectrums_vectors)
        cluster = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold = 0.4)
        labels = cluster.fit_predict(cos_distance)
        wh  = np.where(labels == stats.mode(labels))[0]
        spectrums_select = np.array(spectrums_i)[wh]
        
        mz_list, intensity_list = consensus_spectrum(spectrums_select, mz_window = 0.1)        
        spectrum_consensus = Spectrum(mz=np.array(mz_list),
                                      intensities=np.array(intensity_list),
                                      metadata={'id': 'spectrum_{}'.format(i),
                                                "precursor_mz": mzs[i],
                                                "retention_time": rts[i]})
        spectrums[i] = spectrum_consensus
    return pd.DataFrame({'mz': mzs, 'rt': rts, 'spectrum': spectrums})
    

def consensus_spectrum(spectrums, mz_window = 0.1):
    tot_array = []
    for i, s in enumerate(spectrums):
        mz, intensity = s.peaks.mz, s.peaks.intensities
        array = np.vstack((mz, intensity, np.repeat(i, len(mz)))).T
        tot_array.append(array)
    
    i = 0
    mz, intensity = [], []
    tot_array = np.vstack(tot_array)
    tot_array = tot_array[np.argsort(tot_array[:,0]),:]
    while True:
        if i >= len(tot_array):
            break
        m = tot_array[i,0]
        j = np.searchsorted(tot_array[:,0], m + mz_window)
        a = tot_array[i:j, 0]
        b = tot_array[i:j, 1]
        a = np.round(np.sum(a * b) / np.sum(b), 5)
        b = np.round(np.max(b), 5)
        mz.append(a)
        intensity.append(b)
        i = j
    return mz, intensity


def feature_spectrum_matching(feature_table, spectrums, mz_tol = 0.01, rt_tol = 15):
    print('matching consesus spectrums with features...\n')
    mzs = spectrums.loc[:,'mz'].values
    rts = spectrums.loc[:,'rt'].values
    tandem_ms = []
    for i in tqdm(feature_table.index):
        rt = feature_table.loc[i, 'RT']
        mz = feature_table.loc[i, 'MZ']
        k1 = np.abs(mz - mzs) < mz_tol
        k2 = np.abs(rt - rts) < rt_tol
        kk = np.where(np.logical_and(k1, k2))[0]
        if len(kk) == 0:
            tandem_ms.append(None)
            continue
        elif len(kk) > 1:
            k = kk[np.argmin(np.abs(mz - np.array(mzs)[kk]))]
        else:
            k = kk[0]        
        tandem_ms.append(spectrums.loc[k,'spectrum'])
    feature_table['tandem_ms'] = tandem_ms
    return feature_table
        