# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:02:51 2022

@author: jihon
"""


import AutoMS
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import t

model_path = os.path.join(AutoMS.__path__[0], 'model', 'denoising_autoencoder.pkl')

def evaluate_peaks(peaks, pics, length=14, params=(8.5101, 1.6113, 0.1950), min_width = 6):
    traces = []
    exclude = []

    for i in tqdm(peaks.index):
        rt = peaks.loc[i, 'rt']
        pic = peaks.loc[i, 'pic_label']
        pic = pics[pic]
        
        x = np.linspace(rt - length, rt + length, 50)
        x0, y0 = pic[:,0], pic[:,2]
        y = np.interp(x, x0, y0)
        y = y / np.max(y)
        traces.append(y)
        
        if max(y[24], y[25]) < np.max(y[int(25-min_width/2):int(25+min_width/2)]):
            exclude.append(i)
        elif np.min(y[int(25-min_width/2):int(25+min_width/2)]) < 0.3:
            exclude.append(i)
        else:
            pass
            
    traces = np.array(traces)
    exclude = np.array(exclude)
    
    X = traces
    autoencoder = tf.keras.models.load_model(model_path)
    X_rebuild = autoencoder.predict(X)
    X_rebuild = np.reshape(X_rebuild, [-1, 50])
    
    distance = np.array([np.linalg.norm(X[i] - X_rebuild[i]) for i in range(len(X))])
    worse = np.array([i for i in range(len(distance)) if distance[i] >= params[1]])
    
    scores = t.pdf(distance, params[0], loc = params[1], scale = params[2])
    scores = -np.log10(scores)
    scores[exclude] = 0
    scores[worse] = 0
        
    
    '''
    k = 3233
    print(scores[k])
    y = X[k,:]
    y2 = X_rebuild[k,:]
    
    plt.figure(dpi = 300, figsize = (4.5,3))
    plt.plot(y, lw = 3, label = 'original')
    plt.fill_between(np.arange(50), y, color = 'lightblue', alpha = 0.7)

    plt.plot(y2, lw = 3, color = 'red', label = 'reconstructed')
    plt.fill_between(np.arange(50), y2, color = 'lightpink', alpha = 0.7)
    plt.legend(loc = 'upper left')
    plt.xlabel('scan index')
    plt.ylabel('relative intensity')
    
    plt.figure(dpi = 300)
    y1 = y + np.random.normal(0, 0.1, size = y.shape)
    plt.plot(y1, lw = 3)
    plt.fill_between(np.arange(50), y1, color = 'lightblue')
    '''
    
    return scores
