# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:53:40 2023

@author: DELL
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


class FeatureMatching:
    def __init__(self, peaks):
        self.peaks = peaks
        self.feature_table = None
        
    def simple_matching(self, mz_tol = 0.01, rt_tol = 20):
        peaks = self.peaks
        files = list(peaks.keys())
        rts, mzs, intensities, scores = [], [], [], []
        for f, vals in tqdm(peaks.items()):
            i = files.index(f)
            peak = vals['peaks']
            for j in peak.index:
                k1 = np.abs(peak['mz'][j] - mzs) < mz_tol
                k2 = np.abs(peak['rt'][j] - rts) < rt_tol
                kk = np.where(np.logical_and(k1, k2))[0]
                if len(kk) == 0:
                    mzs.append(peak['mz'][j])
                    rts.append(peak['rt'][j])
                    intensity = np.repeat(np.nan, len(peaks))
                    intensity[i] = peak['intensity'][j]
                    intensities.append(intensity)
                    score = np.repeat(np.nan, len(peaks))
                    if 'score' in peak.columns:
                        score[i] = peak['score'][j]
                    scores.append(score)
                    continue
                elif len(kk) > 1:
                    k = kk[np.argmin(np.abs(peak['mz'][j] - np.array(mzs)[kk]))]
                else:
                    k = kk[0]
                
                n = np.sum(~np.isnan(intensities[k]))
                if np.isnan(intensities[k][i]):
                    mzs[k] = (mzs[k] * n + peak['mz'][j]) / (n+1)
                    rts[k] = (rts[k] * n + peak['rt'][j]) / (n+1)
                    intensities[k][i] = peak['intensity'][j]
                    if 'score' in peak.columns:
                        scores[k][i] = peak['score'][j]
                elif intensities[k][i] < peak['intensity'][j]:
                    mzs[k] = (mzs[k] * n + peak['mz'][j]) / (n+1)
                    rts[k] = (rts[k] * n + peak['rt'][j]) / (n+1)
                    intensities[k][i] = peak['intensity'][j]
                    if 'score' in peak.columns:
                        scores[k][i] = peak['score'][j]
                else:
                    pass
        scores = np.nanmedian(np.array(scores), axis = 1)
        intensities = pd.DataFrame(np.array(intensities), columns=files)    
        output = pd.DataFrame({'RT': rts, 'MZ': mzs, 'Score': scores})
        output = pd.concat([output, intensities], axis = 1)
        output['Annotated Name'] = None
        output['InChIKey'] = None
        output['SMILES'] = None
        output['Matching Score'] = None
        self.feature_table = output
        
        
    def feature_filter(self, min_frac = 0.5):
        if self.feature_table is None:
            return None
        files = list(self.peaks.keys())
        intensities = self.feature_table[files]
        count_nan = np.sum(~np.isnan(intensities), axis = 1)
        wh = np.where(count_nan / intensities.shape[1] >= min_frac)[0]
        if len(wh) > 0:
            self.feature_table = self.feature_table.loc[wh,:]
            self.feature_table = self.feature_table.reset_index(drop = True)
        else:
            self.feature_table = None
        return self.feature_table
        
        