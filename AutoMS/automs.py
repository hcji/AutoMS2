# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:04:55 2022

@author: DELL
"""

import os
import numpy as np
from tqdm import tqdm

from AutoMS import hpic
from AutoMS import peakeval


class AutoMS:
    def __init__(self, data_path, min_intensity):
        """
        Arguments:
            data_path: string
                path to the dataset locally
            min_intensity: string
                minimum intensity of a peak.
        """
        self.data_path = data_path
        self.min_intensity = min_intensity
        self.peaks = None
    
    
    def find_peaks(self, mass_inv = 1, rt_inv = 30, min_snr = 3, max_items = 50000):
        """
        Arguments:
            min_snr: float
                minimum signal noise ratio
            mass_inv: float
                minimum interval of the m/z values
            rt_inv: float
                minimum interval of the retention time
        """
        
        output = {}
        files = os.listdir(self.data_path)
        for i, f in enumerate(files):
            print('processing {}, {}/{} files, set maximum {} ion traces'.format(f, 1+i, len(files), max_items))
            peaks, pics = hpic.hpic(os.path.join(self.data_path, f), 
                                    min_intensity = self.min_intensity, 
                                    min_snr = min_snr, 
                                    mass_inv = mass_inv, 
                                    rt_inv = rt_inv,
                                    max_items = max_items)
            output[f] = {'peaks': peaks, 'pics': pics}
        self.peaks = output
        return output
    
    
    def evaluate_peaks(self):
        if self.peaks is None:
            raise ValueError('Please find peak first')
        for f, vals in self.peaks.items():
            peak = vals['peaks']
            pic = vals['pics']
            score = peakeval.evaluate_peaks(peak, pic)
            self.peaks[f]['peaks']['score'] = score
        return self.peaks
    
    
    
    
    
    
    def save_project(self):
        pass
    






if __name__ == '__main__':
    
    data_path = "E:/Data/Chuanxiong"
    automs = AutoMS(data_path, min_intensity = 10000)
    peaks = automs.find_peaks(max_items = 100000)
    
    
    