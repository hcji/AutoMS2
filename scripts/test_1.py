# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:58:56 2023

@author: DELL
"""


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from AutoMS.automs import AutoMS


path = "E:/Data/Chuanxiong"
files = os.listdir(path)

output = {}
for f in tqdm(files):
    file = path + '/' + f
    peaks = AutoMS(file, min_intensity = 500)
    output[f] = peaks